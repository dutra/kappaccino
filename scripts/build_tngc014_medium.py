#!/usr/bin/env python
"""Medium catalogs of the important images and members of the tngc014 mocks.

    /home/dutra/.conda/envs/lenstronomy/bin/python scripts/build_tngc014_medium.py            # both mocks
    /home/dutra/.conda/envs/lenstronomy/bin/python scripts/build_tngc014_medium.py tngc014_dm

Reads catalogs/<tag>/large/{images,members}.yaml and source/*.csv in burkelenscluster and writes catalogs/<tag>/medium/:
images.yaml, members.yaml, selection.csv, priors.yaml, run.yaml and README.md.  The entries are copied verbatim.

Systems (N_SYSTEMS by class): retain every small system; spectroscopic redshifts only.
Within a class each added system is the farthest from those already taken, in the centroid of its images
(per 30"), its redshift (per 2) and, for folds, the log of the pair separation (per 0.5 dex).  Subhalo systems are
taken by the number of newly covered satellites, then by the number of tied images.

Members (N_MEMBERS): all small members, the BCG, every satellite tied to a selected S system
that is in the member catalog, and then
the members with the largest point-mass deflection M / (pi Sigma_cr D_d^2 d) at any selected image, with M the
Subfind bound mass in the particle types of the lens, d the distance to the image (at least 0.3") and Sigma_cr at
D_ds/D_s = 1.  The ranking uses the truth masses: these are the members that matter, not the ones an observer would
pick from the light.
"""
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

KAPPA = Path(__file__).resolve().parents[1]
GROUPS = Path("/ironwolf/TNGC/output/groups_073/fof_subhalo_tab_073.14.hdf5")
N_SYSTEMS = {"wide": 12, "intermediate": 8, "cusp": 6, "fold": 8, "subhalo": 6}
N_MEMBERS = 100
PTYPES = {"tngc014": [0, 1, 4, 5], "tngc014_dm": [1, 2]}
H = 0.6774


def blocks(path, key):
    """Header lines, and the entries under ``key:`` keyed by name (each a list of lines)."""
    head, items, name = [], {}, None
    inside = False
    for line in path.read_text().splitlines():
        if not inside:
            head.append(line)
            inside = line == f"{key}:"
            continue
        m = re.match(r"  '([^']+)':", line)
        if m:
            name = m.group(1)
            items[name] = []
        items[name].append(line)
    return head, items


def farthest(feat, chosen, n):
    chosen = list(chosen)
    while len(chosen) < min(n, len(feat)):
        d = np.min(np.linalg.norm(feat[:, None] - feat[chosen][None], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return chosen


def select_systems(t, small_ids):
    obs = t[t.in_observed & t.knot.isin(["", "A"])]
    rows = []
    for name, g in obs.groupby("system", sort=False):
        full = t[t.system == name]
        d = np.linalg.norm(g[["x_obs", "y_obs"]].values[:, None] - g[["x_obs", "y_obs"]].values[None], axis=2)
        d[np.diag_indices(len(g))] = np.inf
        subs = sorted({s for v in g.subhalo for s in str(v).split(";") if s})
        rows.append(dict(system=name, cls=g.cls.iloc[0], z=g.z_true.iloc[0], spec=bool(np.all(full.z_catalog == full.z_true)),
                         n=len(g), x=g.x_obs.mean(), y=g.y_obs.mean(), min_sep=d.min(), max_sep=d[np.isfinite(d)].max(),
                         knots=full.knot.nunique(), tied=int((g.subhalo != "").sum()), subs=subs))
    s = pd.DataFrame(rows)
    s = s[s.spec].reset_index(drop=True)
    keep = []
    for cls, n in N_SYSTEMS.items():
        c = s[s.cls == cls].reset_index(drop=True)
        if len(c) < n:
            raise ValueError(f"Too few spectroscopic {cls} systems for quota {n}")
        seeds = [i for i, name in enumerate(c.system) if name in small_ids]
        if len(seeds) > n:
            raise ValueError(f"Small catalog exceeds the {cls} quota")
        if cls == "subhalo":
            order = c.system.iloc[seeds].tolist()
            covered = {s for values in c.subs.iloc[seeds] for s in values}
            for _ in range(n - len(order)):
                left = c[~c.system.isin(order)]
                new = left.subs.map(lambda v: len(set(v) - covered))
                pick = left.assign(new=new).sort_values(["new", "tied"], ascending=False).iloc[0]
                order.append(pick.system)
                covered |= set(pick.subs)
            keep += order
            continue
        feat = np.column_stack([c.x / 30, c.y / 30, c.z / 2] + ([np.log10(c.min_sep) / 0.5] if cls == "fold" else []))
        keep += c.system.iloc[farthest(feat, seeds or [int(c.n.values.argmax())], n)].tolist()
    return s[s.system.isin(keep)].set_index("system").loc[[k for k in s.system if k in keep]].reset_index()


def select_members(tag, t, systems, member_ids, small_ids):
    m = pd.read_csv(KAPPA / f"catalogs/{tag}/source/members_truth.csv", dtype={"id": str})
    with h5py.File(GROUPS, "r") as f:
        mass = f["Subhalo/SubhaloMassType"][:][m["index"].values][:, PTYPES[tag]].sum(axis=1) * 1e10 / H
    with h5py.File(KAPPA / f"data/{tag}/{tag}_snap73_yz_lens_maps.h5", "r") as f:
        te2 = mass / (np.pi * f.attrs["sigma_critical_inf_Msun_pkpc2"] * f.attrs["kpc_per_arcsec"] ** 2)
    im = t[t.in_observed & t.system.isin(systems.system)]
    d = np.hypot(m.x.values[:, None] - im.x_obs.values[None], m.y.values[:, None] - im.y_obs.values[None])
    j = d.argmin(axis=1)
    m["mass_lens"] = mass
    m["theta_e"] = np.sqrt(te2)
    m["nearest_image"] = (im.system.astype(str) + "." + im.image + im.knot.where(im.knot == "", "/" + im.knot)).values[j]
    m["distance"] = d.min(axis=1)
    m["deflection"] = te2 / np.maximum(m.distance, 0.3)
    tied = {s for subs in systems.subs for s in subs}
    m["reason"] = np.where(m.role == "bcg", "bcg", np.where(m.id.isin(tied), "tied to an S system", ""))
    m.loc[m.id.isin(small_ids) & (m.reason == ""), "reason"] = "retained from small"
    forced = m[m.reason != ""]
    if len(forced) > N_MEMBERS:
        raise ValueError("Required members exceed the medium budget")
    rest = m[m.reason == ""].sort_values("deflection", ascending=False).head(N_MEMBERS - len(forced)).assign(reason="deflection")
    out = pd.concat([forced, rest]).sort_values("mag")
    return out, sorted(tied - set(member_ids)), float(m[~m.id.isin(out.id)].deflection.max())


def build(tag):
    src, dest = KAPPA / f"catalogs/{tag}", KAPPA / f"catalogs/{tag}/medium"
    dest.mkdir(exist_ok=True)
    t = pd.read_csv(src / "source/images_truth.csv", keep_default_na=False, na_values=["nan"], dtype={"system": str, "subhalo": str})
    t["in_observed"] = t.in_observed.astype(str) == "True"
    ihead, isys = blocks(src / "large/images.yaml", "systems")
    mhead, mem = blocks(src / "large/members.yaml", "members")
    _, small_systems = blocks(src / "small/images.yaml", "systems")
    _, small_members = blocks(src / "small/members.yaml", "members")
    systems = select_systems(t, set(small_systems))
    members, missing, left = select_members(tag, t, systems, mem, set(small_members))
    assert set(small_systems) <= set(systems.system) <= set(isys)
    assert set(small_members) <= set(members.id) <= set(mem)
    assert len(systems) == sum(N_SYSTEMS.values()) and len(members) == N_MEMBERS
    stamp = "# Medium version by scripts/build_tngc014_medium.py: "
    note = (f"{len(systems)} of {len(isys)} systems, chosen by class for coverage in position and redshift; "
            f"entries copied from ../large/images.yaml.")
    (dest / "images.yaml").write_text("\n".join(
        [ihead[0], stamp + note] + ihead[1:] + [l for k in systems.system for l in isys[k]]) + "\n")
    note = (f"{len(members)} of {len(mem)} members: all small members, the satellites tied to the selected S systems and the "
            f"largest point-mass deflections at the selected images; entries copied from ../large/members.yaml.")
    (dest / "members.yaml").write_text("\n".join(
        [mhead[0], stamp + note] + mhead[1:] + [l for k in members.id for l in mem[k]]) + "\n")
    sel = pd.concat([
        systems.assign(kind="system", id=systems.system, reason=systems.cls, subs=systems.subs.map(";".join))[
            ["kind", "id", "reason", "z", "n", "knots", "min_sep", "max_sep", "tied", "subs"]],
        members.assign(kind="member")[["kind", "id", "reason", "mag", "mass_lens", "theta_e", "nearest_image", "distance", "deflection"]]])
    sel.to_csv(dest / "selection.csv", index=False, float_format="%.5g")
    name = f"{tag}_medium"
    (dest / "priors.yaml").write_text(
        f"# The model of ../large/priors.yaml on the medium catalogs of this folder (README.md).\n"
        f"extends: ../large/priors.yaml\nimages: images.yaml\n\nmembers:\n  members: {{catalog_path: members.yaml}}\n")
    (dest / "run.yaml").write_text(
        f"# How the medium {tag} catalog is fitted: the settings of ../large/run.yaml.\n"
        f"# kappaccino catalogs/{tag}/medium [--profile fast]\n"
        f"extends: ../large/run.yaml\nname: {name}\npriors: priors.yaml\n")
    obs = t[t.in_observed & t.system.isin(systems.system)]
    nim = len(obs[obs.knot.isin(["", "A"])])
    table = "\n".join(f"| {r.system} | {r.cls} | {r.z:.3f} | {r.n} | {r.knots if r.knots > 1 else ''} | {r.min_sep:.2f} | "
                      f"{r.max_sep:.1f} | {', '.join(r.subs)} |" for r in systems.itertuples())
    by = members.reason.value_counts()
    (dest / "README.md").write_text(f"""# {tag}/medium: the important images and members

A reduced version of `../large/images.yaml` and `../large/members.yaml` for quick fits, built by
`scripts/build_tngc014_medium.py`. The entries are copied unchanged, so names, positions, noise
and notes are those of the full catalog. The lens is the same: the members left out are still in the truth maps.

| | large | medium |
|---|---|---|
| systems | {len(isys)} | {len(systems)} |
| observed images (positions with knots) | {len(t[t.in_observed & t.knot.isin(['', 'A'])])} ({int(t.in_observed.sum())}) | {nim} ({len(obs)}) |
| members | {len(mem)} | {len(members)} |

## Systems

Spectroscopic redshifts only, {', '.join(f'{n} {c}' for c, n in N_SYSTEMS.items())}. All 20 small systems are retained. Each added system is the farthest from those taken in image centroid, redshift and,
for folds, pair separation. Added subhalo systems maximize the number of newly covered satellites, then tied images.

| system | class | z | images | knots | closest pair ["] | widest pair ["] | tied satellites |
|---|---|---|---|---|---|---|---|
{table}

## Members

All 50 small members are retained. The selection includes the BCG ({by.get('bcg', 0)}), satellites tied to the selected S systems ({by.get('tied to an S system', 0)}),
other retained small members ({by.get('retained from small', 0)}), and the
{by.get('deflection', 0)} members with the largest point-mass deflection at any selected image: M / (pi Sigma_cr D_d^2 d)
for D_ds/D_s = 1, with M the Subfind bound mass in the particle types of the lens and d the distance to the image.
The smallest value kept is {members[members.reason == 'deflection'].deflection.min():.2f}" and the largest left out is {left:.2f}".
The ranking uses the truth masses, not the light.{' Tied satellites fainter than the member limit, not in the catalog: ' + ', '.join(missing) + '.' if missing else ''}

`selection.csv` lists every selected system and member with these numbers. `priors.yaml` and `run.yaml` extend the
files of the large folder and change only the two catalogs and the run name (`{name}`).
""")
    print(f"{tag}: {len(systems)} systems, {nim} images ({len(obs)} positions), {len(members)} members "
          f"({dict(by)}); smallest kept deflection {members[members.reason == 'deflection'].deflection.min():.2f}, "
          f"largest left out {left:.2f} arcsec; tied satellites not in the member catalog: {missing}")
    print(systems[["system", "cls", "z", "n", "knots", "min_sep", "max_sep"]].to_string(index=False))


if __name__ == "__main__":
    for tag in sys.argv[1:] or PTYPES:
        build(tag)
