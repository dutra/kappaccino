from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

from kappaccino.image_catalog import load_image_catalog, write_image_catalog
from kappaccino.member_catalog import write_member_catalog
from kappaccino.sky import offsets_to_radec


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_master_catalog.py"
spec = importlib.util.spec_from_file_location("build_master_catalog", SCRIPT_PATH)
assert spec is not None
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

REFERENCE = (64.038, -24.067)


def _systems(spec: dict, shift=(0.0, 0.0)) -> dict:
    """{name: (z, notes, [(x, y) arcsec, ...])} -> image-catalog systems, moved by ``shift`` arcsec."""
    out = {}
    for name, (z, notes, points) in spec.items():
        images = {}
        for letter, (x, y) in zip("abcdef", points):
            ra, dec = offsets_to_radec(x + shift[0], y + shift[1], *REFERENCE)
            images[letter] = {"ra": float(ra), "dec": float(dec)}
        out[name] = {"z": z, **({"notes": notes} if notes else {}), "images": images}
    return out


def _grid(n: int, start: float) -> list[tuple[float, float]]:
    return [(start + 7.0 * k, 3.0 * k - 10.0) for k in range(n)]


def _cluster(tmp_path: Path, decisions: list | None = None) -> Path:
    shared = {f"S{k}": (1.0 + 0.1 * k, "spectroscopic z", _grid(3, 11.0 * k)) for k in range(6)}
    base = dict(shared)
    other = {name.replace("S", ""): value for name, value in shared.items()}
    other["90"] = (2.5, "", [(-40.0, 40.0), (-35.0, 42.0)])            # only in the other paper, spectroscopic there
    other["91"] = (0.0, "", [(-60.0, 60.0), (-55.0, 62.0)])            # Lenstool free z
    other["0"] = (1.2, "", shared["S0"][2])                            # S0's z disagrees
    third = {"7": (3.0, "geometric z (lens-model prediction)", [(80.0, -80.0), (85.0, -82.0)])}
    for name, systems, shift in (("base", base, (0, 0)), ("other", other, (0.2, -0.1)), ("third", third, (0, 0))):
        folder = tmp_path / name
        folder.mkdir()
        write_image_catalog(folder / "images.yaml", _systems(systems, shift))
    members = {str(k): {"ra": REFERENCE[0] + 1e-3 * k, "dec": REFERENCE[1], "mag": {"F160W": 20.0}} for k in range(3)}
    write_member_catalog(tmp_path / "base" / "members.yaml", members, band="F160W")
    master = tmp_path / "master"
    master.mkdir()
    (master / "sources.yaml").write_text(yaml.safe_dump({
        "cluster": "test", "base": "base", "match_radius_arcsec": 0.5, "dz_max": 0.01,
        "papers": {"base": {"images": "../base/images.yaml", "members": "../base/members.yaml"},
                   "other": {"images": "../other/images.yaml", "z_kind": "spec", "prefix": "O", "add_systems": "spec"},
                   "third": {"images": "../third/images.yaml", "prefix": "T"}}}))
    (master / "decisions.yaml").write_text(yaml.safe_dump(decisions or []))
    return master


def test_offset_measured_and_policy_applied(tmp_path):
    master = _cluster(tmp_path)
    builder.build(master)
    crossmatch = yaml.safe_load((master / "crossmatch.yaml").read_text())
    dx, dy = crossmatch["frames"]["other"]["applied_arcsec"]
    assert abs(dx - 0.2) < 1e-3 and abs(dy + 0.1) < 1e-3
    images = load_image_catalog(master / "images.yaml", sigma_inst_arcsec=0.05)
    assert sorted(images["system_id"].unique()) == ["O90", "S0", "S1", "S2", "S3", "S4", "S5"]   # 91 free z, T7 model z
    assert crossmatch["conflicts"] == [{"system": "S0", "kind": "z", "values": {"other:0": 1.2, "master": 1.0}}]
    gold = load_image_catalog(master / "images_gold.yaml", sigma_inst_arcsec=0.05)
    assert "O90" not in set(gold["system_id"]) and "S3" in set(gold["system_id"])
    # O90 is moved onto the base frame
    ra, dec = offsets_to_radec(-40.0, 40.0, *REFERENCE)
    row = images[images["image_label"] == "O90.a"].iloc[0]
    assert abs(row["ra"] - float(ra)) < 1e-6 and abs(row["dec"] - float(dec)) < 1e-6


def test_decisions(tmp_path):
    master = _cluster(tmp_path, [{"set_z": "S0", "z": 1.2, "reason": "test"}, {"exclude": "S5", "reason": "test"},
                                 {"include": "third:7", "reason": "test"}])
    builder.build(master)
    crossmatch = yaml.safe_load((master / "crossmatch.yaml").read_text())
    images = load_image_catalog(master / "images.yaml", sigma_inst_arcsec=0.05)
    assert crossmatch["conflicts"] == []
    assert images.loc[images["system_id"] == "S0", "catalog_z"].iloc[0] == 1.2
    assert "S5" not in set(images["system_id"]) and "T7" in set(images["system_id"])
