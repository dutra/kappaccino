"""priors.yaml and run.yaml: the value syntax, the loaders, and the cluster folders of catalogs/."""
import math
from pathlib import Path

import numpyro.distributions as dist
import pytest

from kappaccino import priors
from kappaccino.config import DPIEHaloConfig, ShearHaloConfig
from kappaccino.run_files import input_files, load_priors, load_run, parse_value

ROOT = Path(__file__).resolve().parents[1]
CLUSTERS = {"catalogs/hera": ("hera", ["fast"]), "catalogs/ares": ("ares", ["fast"]),
            "catalogs/as1063/bergamini2019": ("as1063", ["fast", "jwst"]),
            "catalogs/as1063/diego2026": ("as1063_jwst", ["fast", "hst"]),
            "catalogs/m0416/bergamini2023": ("m0416_bergamini2023", ["fast", "jwst"]),
            "catalogs/m0416/canucs2025": ("m0416_canucs2025", ["fast", "jwst"])}


def same(a, b):
    return priors.to_dict(a) == pytest.approx(priors.to_dict(b)) and priors.to_dict(a)["family"] == priors.to_dict(b)["family"]


def test_numbers_and_strings():
    assert parse_value("2.3 / 0.72") == pytest.approx(2.3 / 0.72)
    assert parse_value("3 * arcsec", {"arcsec": 5.0}) == 15.0
    assert parse_value("(5.0 * 100.0) ** 0.5") == pytest.approx(500 ** 0.5)
    assert parse_value("1e-3") == 1e-3
    assert parse_value("inf") == math.inf
    for text in ("auto", "student_t", "F160W", "system:K1", "_jwst", "smoothed-local-image-plane", "image:K34.b"):
        assert parse_value(text) == text


def test_prior_functions():
    assert same(parse_value("uniform(-2, 4)"), dist.Uniform(-2.0, 4.0))
    assert same(parse_value("loguniform(5, 100)"), dist.LogUniform(5.0, 100.0))
    assert same(parse_value("normal(0.25, 0.05) in [0.1, 0.4]"), dist.TruncatedNormal(0.25, 0.05, low=0.1, high=0.4))
    assert same(parse_value("normal(0.0, 1.0)"), dist.Normal(0.0, 1.0))
    assert same(parse_value("student_t(4, 0, 1)"), dist.StudentT(4.0, 0.0, 1.0))
    assert same(parse_value("lognormal(20, 1) in [0.5, 80]"), priors.truncated_lognormal(20.0, 1.0, low=0.5, high=80.0))
    assert same(parse_value("lognormal(median=0.08, width=0.5)"), priors.truncated_lognormal(0.08, 0.5))
    assert same(parse_value("lognormal(mean=1200, std=200) in [1000, inf]"), priors.lognormal_from_moments(1200.0, 200.0, low=1000.0))
    assert same(parse_value("lognormal(mean=500, std=500) in [0, inf]"), priors.lognormal_from_moments(500.0, 500.0))


@pytest.mark.parametrize("text", ["uniform(1, 2) in [0, 3]", "lognormal(mean=1, sd=2)", "normal(0, 1) in [2, 1]",
                                  "lognormal(1)", "normal(0, x)"])
def test_malformed_priors_raise(text):
    with pytest.raises(ValueError):
        parse_value(text)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_halos_presets_starts_and_shapes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "c/images.yaml", "systems: {}\n")
    path = write(tmp_path / "c/priors.yaml", """
z_lens: 0.5
images: images.yaml
halos:
  a: {preset: cluster_scale, x_centre: 1.0, y_centre: 2.0, v_disp: 900.0, core_radius_kpc: 12.0}
  b:
    x_centre: normal(0.0, 1.5) in [-6, 6]
    y_centre:
      start: 1.0
      prior: uniform(-2, 4)
    shape: {ellipticity: [0.2, 0.94], angle: [100, 180], soft_cap: 0.9}
    core_radius_kpc: 2 * arcsec
    cut_radius_kpc: 1500
    v_disp: 600
    cut_excess_kpc: loguniform(5, 100)
  s: {profile: shear, preset: free, gamma_upper: 0.2}
scaling:
  softening_length_prior: lognormal(3.0, 0.15)
""")
    model, scaling = load_priors(path)
    a, b, s = model.large_halos
    reference = DPIEHaloConfig.cluster_scale("a", 1.0, 2.0, 0.5, v_disp=900.0, softening_kpc=3.0)
    assert a.core_radius_kpc == 12.0 and "core_radius_kpc" not in a.priors          # a number fixes a preset's prior
    assert set(a.priors) == set(reference.priors) - {"core_radius_kpc"}
    assert same(a.priors["cut_excess_kpc"], reference.priors["cut_excess_kpc"])      # softening from the scaling prior
    assert b.x_centre == 0.0 and b.y_centre == 1.0
    assert b.ellipticite == pytest.approx(0.55) and b.angle_pos == 140.0 and b.ellipticity_soft_cap == 0.9
    assert set(b.priors) == {"x_centre", "y_centre", "e1", "e2", "cut_excess_kpc"}
    assert b.core_radius_kpc > 2.0                                                  # arcsec in kpc at z = 0.5
    assert isinstance(s, ShearHaloConfig) and same(s.priors["gamma1"], ShearHaloConfig.free("s", 0.5, gamma_upper=0.2).priors["gamma1"])
    assert model.image_constraints.catalog_path == "c/images.yaml"                  # relative to the file, then to cwd


@pytest.mark.parametrize("entry, message", [
    ("{x_centre: 0, y_centre: 0, ellipticite: 0, angle_pos: 0, core_radius_kpc: 1, cut_radius_kpc: 10, v_disp: 100, sigma: 3}", "unknown key"),
    ("{preset: free}", "preset"),
    ("{x_centre: 0, y_centre: 0, ellipticite: 0, angle_pos: 0, core_radius_kpc: 1, cut_radius_kpc: 10, v_disp: 100, e1: 0.1}", "sampled only"),
    ("{x_centre: 0, y_centre: 0, ellipticite: 'uniform(0, 1)', angle_pos: 0, core_radius_kpc: 1, cut_radius_kpc: 10, v_disp: 100}", "cannot be sampled"),
])
def test_bad_halos_raise(tmp_path, entry, message):
    path = write(tmp_path / "priors.yaml", f"z_lens: 0.5\nhalos:\n  h: {entry}\n")
    with pytest.raises(ValueError, match=message):
        load_priors(path)


def test_unknown_top_level_keys_raise(tmp_path):
    with pytest.raises(ValueError, match="unknown keys"):
        load_priors(write(tmp_path / "priors.yaml", "z_lens: 0.5\nformat: 1\n"))
    write(tmp_path / "priors.yaml", "z_lens: 0.5\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_run(write(tmp_path / "run.yaml", "priors: priors.yaml\nsamplng: {warmup: 1}\n"))
    with pytest.raises(TypeError):
        load_run(write(tmp_path / "run.yaml", "priors: priors.yaml\nsampling: {warmpu: 1}\n"))


@pytest.mark.parametrize("folder", sorted(CLUSTERS))
def test_cluster_folders_load_and_validate(folder, monkeypatch):
    monkeypatch.chdir(ROOT)
    name, profiles = CLUSTERS[folder]
    for chosen in ([], *([p] for p in profiles)):
        config = load_run(folder, chosen, ["paths.run_name=test"]).validate()
        assert config.paths.figure_prefix == name
        assert config.model.image_constraints.catalog_path.startswith(folder)
        assert config.reporting.extra_images == ("fast" not in chosen)


def test_profiles_extends_and_overrides(monkeypatch):
    monkeypatch.chdir(ROOT)
    base = load_run("catalogs/m0416/canucs2025")
    fast = load_run("catalogs/m0416/canucs2025", ["fast"], ["sampling.warmup=1000", "likelihood.lambda_b_prior=lognormal(10, 1) in [0.5, 80]"])
    # extends ../bergamini2023/run.yaml: its sections and profiles, this folder's priors.
    assert base.members.far_field_cut_radii == 2.0 and base.image_catalog.paper_fold_cutouts[0] == "system:K1"
    assert base.image_catalog.candidate_catalog_path == "src/kappaccino/data/m0416_candidates.json"
    assert base.scaling.member_shape_scatter == 0.1 and fast.scaling.member_shape_scatter is None
    assert base.likelihood.gp_blocks == "all" and fast.likelihood.gp_blocks == "system" and fast.likelihood.gp_knee_project_modes == ()
    assert fast.sampling.warmup == 1000 and fast.sampling.max_tree_depth == 8
    assert same(fast.likelihood.lambda_b_prior, priors.truncated_lognormal(10.0, 1.0, low=0.5, high=80.0))
    with pytest.raises(ValueError, match="no profile"):
        load_run("catalogs/hera", ["slow"])


def test_input_files_follow_extends(monkeypatch):
    monkeypatch.chdir(ROOT)
    files = [str(p) for p in input_files("catalogs/as1063/diego2026")]
    assert files == ["catalogs/as1063/diego2026/run.yaml", "catalogs/as1063/bergamini2019/run.yaml",
                     "catalogs/as1063/diego2026/priors.yaml", "catalogs/as1063/bergamini2019/priors.yaml"]


def test_command_line(monkeypatch, capsys):
    from kappaccino import cli
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit):
        cli.main(["catalogs/nowhere"])
    assert "no run.yaml" in capsys.readouterr().err
    config = cli.build_config("catalogs/as1063/diego2026", profiles=["fast"], overrides=["sampling.warmup=7"], campaign="c", run_label="r")
    assert config.paths.output_dir == "results/c/as1063_jwst" and config.paths.run_name == "r" and config.sampling.warmup == 7
    assert cli.main(["catalogs/hera", "--profile", "fast", "--dry-run"]) == 0
    assert '"dry_run": true' in capsys.readouterr().out
    assert cli.build_config("catalogs/hera").paths.output_dir == f"results/{cli.today()}/hera"
    seen = {}
    monkeypatch.setattr(cli, "build_config", lambda *a, **k: seen.update(k) or (_ for _ in ()).throw(SystemExit(0)))
    with pytest.raises(SystemExit):
        cli.main(["catalogs/hera", "--warmup", "100", "--samples", "50", "--tree-depth", "9", "--target-accept", "0.9",
                  "--chains", "2", "--seed", "7", "--starts", "16", "--warm-start", "results/a/hera/r", "--draws", "8",
                  "--skip-plots", "--set", "sampling.samples=60", "--dry-run"])
    assert seen["overrides"] == ["runtime.chains=2", "runtime.seed=7", "search.n_starts=16", "search.warm_start=results/a/hera/r",
                                 "sampling.warmup=100", "sampling.samples=50", "sampling.max_tree_depth=9",
                                 "sampling.target_accept=0.9", "reporting.draws=8", "reporting.image_draws=8",
                                 "runtime.skip_plots=true", "sampling.samples=60"]
    with pytest.raises(SystemExit):
        cli.main(["catalogs/hera", "--target-accept", "1.5"])
    monkeypatch.undo()
    monkeypatch.chdir(ROOT)
    built = cli.build_config("catalogs/hera", overrides=seen["overrides"])
    assert (built.sampling.warmup, built.sampling.samples, built.sampling.max_tree_depth, built.sampling.target_accept) == (100, 60, 9, 0.9)
    assert (built.runtime.chains, built.runtime.seed, built.runtime.skip_plots, built.search.n_starts) == (2, 7, True, 16)
    assert built.search.warm_start == "results/a/hera/r" and (built.reporting.draws, built.reporting.image_draws) == (8, 8)
    swapped = cli.build_config("catalogs/hera", images="catalogs/ares/images.yaml", overrides=["partition.catalog_path=u.csv"])
    assert swapped.model.image_constraints.catalog_path == "catalogs/ares/images.yaml" and swapped.partition.catalog_path == "u.csv"
