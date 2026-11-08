from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

from kappaccino.io.image_catalog import load_image_catalog, write_image_catalog
from kappaccino.io.member_catalog import load_member_catalog, write_member_catalog
from kappaccino.io.sky import offsets_to_radec


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_master_catalog.py"
spec = importlib.util.spec_from_file_location("build_master_catalog", SCRIPT_PATH)
assert spec is not None
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

REFERENCE = (64.038, -24.067)


def _systems(spec: dict, shift=(0.0, 0.0)) -> dict:
    """{name: (z, notes, [(x, y) arcsec, ...], image notes or None)} -> image-catalog systems, moved by ``shift`` arcsec."""
    out = {}
    for name, (z, notes, points, image_notes) in spec.items():
        images = {}
        for letter, (x, y) in zip("abcdef", points):
            ra, dec = offsets_to_radec(x + shift[0], y + shift[1], *REFERENCE)
            images[letter] = {"ra": float(ra), "dec": float(dec), **({"notes": image_notes} if image_notes else {})}
        out[name] = {"z": z, **({"notes": notes} if notes else {}), "images": images}
    return out


def _grid(n: int, start: float) -> list[tuple[float, float]]:
    return [(start + 7.0 * k, 3.0 * k - 10.0) for k in range(n)]


def _cluster(tmp_path: Path, decisions: list | None = None) -> Path:
    shared = {f"S{k}": (1.0 + 0.1 * k, "spectroscopic z", _grid(3, 11.0 * k), None) for k in range(6)}
    base = dict(shared)
    base["S6"] = (2.0, "", _grid(2, 70.0), None)                                    # base only, z not spectroscopic
    other = {name.replace("S", ""): value for name, value in shared.items()}
    other["0"] = (1.2, "", shared["S0"][2] + [(5.0, 5.0)], None)                   # S0's z disagrees; an extra image
    other["90"] = (2.5, "", [(-40.0, 40.0), (-35.0, 42.0)], "class copper")        # other paper only, low quality
    other["91"] = (0.0, "", [(-60.0, 60.0), (-55.0, 62.0)], None)                  # Lenstool free z
    third = {"7": (3.0, "geometric z (lens-model prediction)", [(80.0, -80.0), (85.0, -82.0)], None)}
    for name, systems, shift in (("base", base, (0, 0)), ("other", other, (0.2, -0.1)), ("third", third, (0, 0))):
        folder = tmp_path / name
        folder.mkdir()
        write_image_catalog(folder / "images.yaml", _systems(systems, shift))
    base_members = {str(k): {"ra": REFERENCE[0] + 1e-3 * k, "dec": REFERENCE[1], "mag": {"F160W": 20.0}} for k in range(3)}
    base_members["0"]["z"] = {"value": 0.40, "kind": "spec"}                       # a spectroscopic member
    base_members["1"]["z"] = {"value": 0.90, "kind": "spec"}                       # a spectroscopic interloper
    write_member_catalog(tmp_path / "base" / "members.yaml", base_members, band="F160W")
    other_members = {"2": {"ra": REFERENCE[0] + 2e-3, "dec": REFERENCE[1], "mag": {"F160W": 20.0}},
                     "9": {"ra": REFERENCE[0] + 9e-3, "dec": REFERENCE[1], "mag": {"F160W": 21.0}}}
    write_member_catalog(tmp_path / "other" / "members.yaml", other_members, band="F160W")
    master = tmp_path / "master"
    master.mkdir()
    (master / "sources.yaml").write_text(yaml.safe_dump({
        "cluster": "test", "base": "base", "match_radius_arcsec": 0.5, "dz_max": 0.01, "z_cluster": 0.4,
        "papers": {"base": {"images": "../base/images.yaml", "members": "../base/members.yaml"},
                   "other": {"images": "../other/images.yaml", "members": "../other/members.yaml", "z_kind": "spec", "prefix": "O",
                             "quality": {"class copper": 0.3}},
                   "third": {"images": "../third/images.yaml", "prefix": "T"}}}))
    (master / "decisions.yaml").write_text(yaml.safe_dump(decisions or []))
    return master


def _load(master: Path):
    crossmatch = yaml.safe_load((master / "crossmatch.yaml").read_text())
    return crossmatch, load_image_catalog(master / "images.yaml", sigma_inst_arcsec=0.05)


def test_nothing_dropped_and_tiers(tmp_path):
    master = _cluster(tmp_path)
    builder.build(master)
    crossmatch, images = _load(master)
    dx, dy = crossmatch["frames"]["other"]["applied_arcsec"]
    assert abs(dx - 0.2) < 1e-3 and abs(dy + 0.1) < 1e-3
    assert sorted(images.system_id.unique()) == ["O90", "O91", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "T7"]
    tiers = {name: entry["tier"] for name, entry in crossmatch["systems"].items()}
    assert tiers["S1"] == "diamond"                  # spectroscopic, two papers
    assert tiers["S6"] == "bronze"                   # one paper, z of unknown kind
    assert tiers["O90"] == "copper"                  # one paper, low-quality images
    assert crossmatch["free_z"] == ["O91"]
    assert crossmatch["conflicts"] == [{"system": "S0", "kind": "z", "values": {"other:0": 1.2, "master": 1.0}}]
    by_label = images.set_index("image_label")
    assert by_label.loc["S1.a", "association_prior"] == 1.0
    assert by_label.loc["O90.a", "association_prior"] == round(0.15 * 0.3, 4)
    assert by_label.loc["O91.a", "catalog_z"] == 0.0
    # the image the other paper adds to S0 is a candidate, at half the system's reliability, on the base frame
    candidate = by_label.loc["S0.d"]
    assert candidate.association_prior == 0.5 and crossmatch["candidate_images"][0]["image"] == "S0.d"
    ra, dec = offsets_to_radec(5.0, 5.0, *REFERENCE)
    assert abs(candidate.ra - float(ra)) < 1e-6 and abs(candidate.dec - float(dec)) < 1e-6
    # members: spectroscopic member 1, interloper kept with a low prior, two papers 0.9, the other paper's own member added
    members = load_member_catalog(master / "members.yaml", (0, 0.0, 0.0)).set_index("id")
    assert members.loc["0", "catalog_p_member"] == 1.0
    assert members.loc["1", "catalog_p_member"] == 0.02
    assert members.loc["2", "catalog_p_member"] == 0.9
    assert members.loc["O9", "catalog_p_member"] == 0.7


def test_decisions(tmp_path):
    master = _cluster(tmp_path, [{"set_z": "S0", "z": 1.2, "reason": "test"}, {"exclude": "S5", "reason": "test"},
                                 {"set_tier": "S6", "tier": "gold", "reason": "test"},
                                 {"link": ["base:S6", "third:7"], "reason": "test"}])
    builder.build(master)
    crossmatch, images = _load(master)
    assert crossmatch["conflicts"] == []
    assert images.loc[images.system_id == "S0", "catalog_z"].iloc[0] == 1.2
    assert "S5" not in set(images.system_id)
    assert crossmatch["systems"]["S6"]["papers"] == {"base": ["S6"], "third": ["7"]}
    assert crossmatch["systems"]["S6"]["tier"] == "gold"
    assert "T7" not in set(images.system_id)          # linked into S6
