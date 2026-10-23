$XONSH_SHOW_TRACEBACK = True

import json
import os
import sys
import time
from pathlib import Path

cores = 4  # NUTS chains; one JAX CPU device per chain.
seed = 12345

$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataclasses import replace

from lenscluster.config import (
    CosmologyConfig,
    DPIEHaloConfig,
    ImageCatalogCutoutConfig,
    ImageConstraintsConfig,
    LensClusterSolverConfig,
    LensModelConfig,
    MemberPopulationConfig,
    PriorConfig,
    RGBDisplayConfig,
    ReferenceFrameConfig,
    RunPathsConfig,
    RuntimeConfig,
    SamplingConfig,
    ScalingModelConfig,
    SearchConfig,
    ShearHaloConfig,
    TruthRecoveryConfig,
)
from lenscluster.cpu_runtime import configure_cpu_runtime, resolve_cpu_plan
from lenscluster.jax_cosmology import kpc_per_arcsec_from_config

OUTPUT_DIR_LABEL = "oct22_single_stage"

# HFF display settings for observed clusters.
HFF_RGB_BANDS = ["F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W"]
HFF_RGB_DISPLAY = {"q": 6.4, "stretch": 0.0145, "minimum": -5.5e-4, "red_gain": 0.47, "green_gain": 0.91, "blue_gain": 3.95}

VALID_CLUSTERS = "ARES, HERA, AS1063"
FF_RGB_BANDS = ("F435W", "F606W", "F814W")
FF_RGB_DISPLAY = RGBDisplayConfig(q=6.8, stretch=0.0158, minimum=0.00105, red_gain=0.62, green_gain=0.78, blue_gain=3.65)


def _usage() -> str:
    return (
        "Usage: xonsh run.xsh <cluster> [--campaign NAME] [--run-label NAME] [--image-metadata PATH] [--init hybrid|multistart|staged] [--search lbfgs|adam|svi] [--stage init|final|both] [--fast] [--dry-run]\n"
        "  --fast: exploratory settings: members deflecting every image by < 0.03\" are dropped, far-field members allowed down to\n"
        "          2 cut radii from an image, classification at the current model, 16 search starts, 300+150 draws at tree depth 7\n"
        "          (check the audit's far-field error before trusting a fast run)\n"
        f"Valid clusters: {VALID_CLUSTERS}"
    )


def _parse_args(argv: list[str]) -> dict:
    if len(argv) == 2 and argv[1] in {"--help", "-h"}:
        print(_usage())
        raise SystemExit(0)
    if len(argv) < 2:
        print(_usage())
        raise SystemExit(2)
    cluster = str(argv[1]).strip().upper()
    values = {"campaign": OUTPUT_DIR_LABEL, "run_label": None, "image_metadata": None, "dry_run": False, "fast": False, "search": "lbfgs", "stage": "both", "init": "multistart"}
    flag_to_key = {"--campaign": "campaign", "--run-label": "run_label", "--image-metadata": "image_metadata", "--search": "search", "--stage": "stage", "--init": "init"}
    index = 2
    while index < len(argv):
        flag = argv[index]
        if flag in {"--dry-run", "--fast"}:
            values[flag.lstrip("-").replace("-", "_")] = True
            index += 1
            continue
        if flag in {"--help", "-h"}:
            print(_usage())
            raise SystemExit(0)
        if flag not in flag_to_key or index + 1 >= len(argv):
            print(f"Unknown argument or missing value: {flag!r}\n{_usage()}")
            raise SystemExit(2)
        values[flag_to_key[flag]] = argv[index + 1]
        index += 2
    for key in ("campaign", "run_label"):
        value = values[key]
        if value is not None and (not str(value).strip() or str(value) in {".", ".."} or "/" in str(value) or "\\" in str(value)):
            print(f"--{key.replace('_', '-')} must be a nonempty single path segment.")
            raise SystemExit(2)
    if values["search"] not in {"lbfgs", "adam", "svi"}:
        print("--search must be lbfgs, adam or svi.")
        raise SystemExit(2)
    if values["init"] not in {"hybrid", "multistart", "staged"}:
        print("--init must be hybrid, multistart or staged.")
        raise SystemExit(2)
    if values["init"] == "staged" and values["search"] == "lbfgs":
        values["search"] = "adam"
    if values["stage"] not in {"init", "final", "both"}:
        print("--stage must be init, final or both.")
        raise SystemExit(2)
    return dict(cluster=cluster, campaign_name=str(values["campaign"]).strip(), run_label=values["run_label"],
                dry_run=values["dry_run"], fast=values["fast"], image_metadata=values["image_metadata"], search_method=values["search"], stage=values["stage"], init_strategy=values["init"])


# The solver's tuned defaults live in lenscluster.config; a cluster entry carries only what is specific to it: the
# image catalog, the cosmology, where the cluster-scale halos sit, the member catalog, the truth maps and the
# simulation's softening.  Both simulated clusters are at z = 0.5 with F160W magnitudes, where L* is m_F160W = 19.82.
FF_SIMS_CLUSTERS = {
    "ARES": {
        "cluster_key": "ares",
        "image_catalog": "data/ff_sims/ares/ares_obs_arcs.cat",
        "cosmology": CosmologyConfig(H0=70.4, Om0=0.272, Ode0=0.728),
        # One cluster-scale halo at the light peak of each clump of the MOKA cluster (Meneghetti et al. 2017) plus shear.
        "halos": (
            DPIEHaloConfig.cluster_scale("1", 20.0, -32.0, 0.5, v_disp=950.0),
            DPIEHaloConfig.cluster_scale("2", -40.0, 40.0, 0.5, v_disp=950.0),
            ShearHaloConfig.free("S1", 0.5),
        ),
        "member_population": MemberPopulationConfig.literature("potfile_1", "data/ff_sims/ares/ares_cluster_members_potfile.cat", 0.5, prior_mag0=19.82),
        # The brightest cluster galaxies stay in the scaling relation, as on Hera: independent halos cost parameters, made
        # the chains multi-well and did not improve the fit.
        "independent_member_halos": (),
        "kappa_true_fits": "data/ff_sims/published/ares/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/published/ares/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/published/ares/gammay_z9_0.fits",
        # Ares is semi-analytic: its subhalos are analytic profiles, so there is no resolution softening to match.
        "softening_length_kpc": 0.0,
    },
    "HERA": {
        "cluster_key": "hera",
        "image_catalog": "data/ff_sims/hera/hera_obs_arcs.cat",
        "cosmology": CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
        "halos": (
            DPIEHaloConfig.cluster_scale("1", 19.54080009, 2.37820005, 0.507, v_disp=800.0),
            DPIEHaloConfig.cluster_scale("2", 0.001, 0.003, 0.507, v_disp=700.0),
            ShearHaloConfig.free("S1", 0.507),
        ),
        "member_population": MemberPopulationConfig.literature("potfile_1", "data/ff_sims/hera/hera_cluster_members_potfile.cat", 0.507, prior_mag0=19.82),
        "independent_member_halos": (),
        "kappa_true_fits": "data/ff_sims/published/hera/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/published/hera/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/published/hera/gammay_z9_0.fits",
        # Hera is an N-body cluster (H0 = 72, softening 2.3 h^-1 kpc = 3.19 kpc): added in quadrature to the member cores so
        # that their cusps match the simulation's resolution; the dPIE total mass is independent of the core.
        "softening_length_kpc": 2.3 / 0.72,
    },
}


# Native configuration transcribed from
# data/Bergamini/RXJ2248_Bergamini19/Bergamini_RXCJ2248_lenstool_Normal.par.
# That model gives radii in arcseconds; native configurations require kpc.
def _hff_clusters():
    AS1063_KPC_PER_ARCSEC = kpc_per_arcsec_from_config(0.348, {"H0": 70.0, "Om0": 0.3})
    AS1063_CATALOG_DIR = "data/Bergamini/RXJ2248_Bergamini19"
    return {
        "AS1063": {
            "cluster_key": "as1063",
            "reference": ReferenceFrameConfig(reference=3, ra0_deg=342.18321, dec0_deg=-44.530878),
            "image_catalog": f"{AS1063_CATALOG_DIR}/obs_arcs.dat",
            "cosmology": CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7),
            "halos": (
                DPIEHaloConfig(
                    id="1", x_centre=0.0, y_centre=0.0,
                    ellipticite=0.5, angle_pos=-39.0,
                    core_radius_kpc=13.5871 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=2000.0 * AS1063_KPC_PER_ARCSEC,
                    v_disp=1500.0, z_lens=0.348,
                    priors={
                        "x_centre": PriorConfig("uniform", lower=-2.0, upper=4.0, step=0.1),
                        "y_centre": PriorConfig("uniform", lower=-2.0, upper=2.0, step=0.1),
                        "ellipticite": PriorConfig("uniform", lower=0.4, upper=0.8, step=0.05),
                        # Use an initial angle inside the published prior bounds.
                        "angle_pos": PriorConfig("uniform", lower=-42.0, upper=-36.0, step=0.1),
                        "core_radius_kpc": PriorConfig(
                            "uniform", lower=5.0 * AS1063_KPC_PER_ARCSEC,
                            upper=35.0 * AS1063_KPC_PER_ARCSEC, step=0.1 * AS1063_KPC_PER_ARCSEC,
                        ),
                        "v_disp": PriorConfig("truncated_normal", mean=1200.0, std=200.0, lower=1000.0, upper=float("inf")),
                    },
                ),
                DPIEHaloConfig(
                    id="2", x_centre=-84.6455835, y_centre=45.01476,
                    ellipticite=0.0, angle_pos=0.0,
                    core_radius_kpc=0.05 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=2000.0 * AS1063_KPC_PER_ARCSEC,
                    v_disp=300.0, z_lens=0.348,
                    priors={
                        "x_centre": PriorConfig("uniform", lower=-100.0, upper=50.0, step=0.1),
                        "y_centre": PriorConfig("uniform", lower=-50.0, upper=70.0, step=0.1),
                        "v_disp": PriorConfig("truncated_normal", mean=500.0, std=500.0, lower=0.0, upper=float("inf")),
                    },
                ),
                # Fixed gas components from the same model.
                DPIEHaloConfig(
                    id="3", x_centre=18.898068688515124, y_centre=-73.3566238994408,
                    ellipticite=0.804756331682532, angle_pos=-162.04793432346537,
                    core_radius_kpc=188.39998560000458 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=189.2403216000046 * AS1063_KPC_PER_ARCSEC,
                    v_disp=355.9263730614334, z_lens=0.348,
                ),
                DPIEHaloConfig(
                    id="4", x_centre=-18.0530993114858, y_centre=13.469568100561368,
                    ellipticite=0.13273356095824723, angle_pos=-27.795099374268442,
                    core_radius_kpc=36.323966400000884 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=339.1572480000082 * AS1063_KPC_PER_ARCSEC,
                    v_disp=442.5672747534171, z_lens=0.348,
                ),
                DPIEHaloConfig(
                    id="5", x_centre=0.19813268851464938, y_centre=-1.2372958994389047,
                    ellipticite=0.34435532608684766, angle_pos=-15.486418948811508,
                    core_radius_kpc=14.434532160000352 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=356.49532800000867 * AS1063_KPC_PER_ARCSEC,
                    v_disp=249.68220495795586, z_lens=0.348,
                ),
            ),
            "member_population": MemberPopulationConfig(
                id="potfile_1", catalog_path=f"{AS1063_CATALOG_DIR}/members.cat",
                mag0="auto", prior_mag0=16.176001, corekpc=0.05 * AS1063_KPC_PER_ARCSEC,
                sigma=310.0, cutkpc=10.0 * AS1063_KPC_PER_ARCSEC, z_lens=0.348,
                # Member-scaling priors require finite bounds; cap the Gaussian tails at +10 sigma.
                sigma_prior=PriorConfig("truncated_normal", mean=310.0, std=100.0, lower=0.0, upper=1310.0),
                cutkpc_prior=PriorConfig(
                    "truncated_normal", mean=10.0 * AS1063_KPC_PER_ARCSEC,
                    std=10.0 * AS1063_KPC_PER_ARCSEC, lower=0.0, upper=110.0 * AS1063_KPC_PER_ARCSEC,
                ),
            ),
            "independent_member_halos": (),
            "softening_length_kpc": 0.0,
            "softening_length_prior_log_sigma": 0.15,
        },
    }


CLUSTERS = set(FF_SIMS_CLUSTERS) | {"AS1063"}


def build_runtime(cores: int) -> RuntimeConfig:
    return RuntimeConfig(
        seed=seed,
        chains=cores,
        resume=True,
        numpyro_print_summary=True,
        nuts_chain_method="parallel",
        cpu_threads_per_chain=2,  # Shared CPU budget = concurrent chains * this value; None disables tuning.
    )


def build_config(
    cluster: str,
    *,
    cores: int,
    campaign_name: str,
    run_label: str | None = None,
    image_metadata: str | None = None,
    runtime: RuntimeConfig | None = None,
    search_method: str = "lbfgs",
    init_strategy: str = "multistart",
    fast: bool = False,
) -> LensClusterSolverConfig:
    """The solver configuration for one cluster: library defaults plus the cluster entry, the command-line choices and,
    with ``fast``, the exploratory overrides."""
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be one of {', '.join(sorted(CLUSTERS))}; got {cluster!r}.")
    is_hff = cluster not in FF_SIMS_CLUSTERS
    cluster_config = _hff_clusters()[cluster] if is_hff else FF_SIMS_CLUSTERS[cluster]
    config = LensClusterSolverConfig(
        model=LensModelConfig(
            reference=cluster_config.get("reference", ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0)),
            cosmology=cluster_config["cosmology"],
            large_halos=cluster_config["halos"],
            independent_member_halos=cluster_config["independent_member_halos"],
            member_populations=(cluster_config["member_population"],),
            image_constraints=ImageConstraintsConfig(catalog_path=cluster_config["image_catalog"], metadata_path=image_metadata),
        ),
        paths=RunPathsConfig(output_dir=f"results/{campaign_name}/{cluster_config['cluster_key']}",
                             run_name=run_label if run_label is not None else "smoothed-local-image-plane",
                             figure_prefix=cluster_config["cluster_key"]),
        runtime=runtime if runtime is not None else build_runtime(cores),
        search=SearchConfig(strategy=init_strategy, method=search_method, staged_method=search_method if search_method != "lbfgs" else "adam"),
        sampling=SamplingConfig(warmup=1000, samples=1000, max_tree_depth=8, target_accept=0.85),
        scaling=ScalingModelConfig(softening_length_kpc=float(cluster_config.get("softening_length_kpc", 0.0))),
        truth=TruthRecoveryConfig(kappa_true_fits=cluster_config.get("kappa_true_fits"), gammax_true_fits=cluster_config.get("gammax_true_fits"),
                                  gammay_true_fits=cluster_config.get("gammay_true_fits")),
        image_catalog=ImageCatalogCutoutConfig(
            image_dir="data/BUFFALO_Images" if is_hff else "data/ff_sims",
            image_scale="auto",
            bands=tuple(HFF_RGB_BANDS) if is_hff else FF_RGB_BANDS,
            rgb=RGBDisplayConfig(**HFF_RGB_DISPLAY) if is_hff else FF_RGB_DISPLAY,
        ),
    )
    if fast:
        # Exploratory: members deflecting every image by < 0.03" leave the model (Ares: 332 -> ~150), far-field members
        # allowed down to 2 cut radii, classification at the current model, fewer starts, shorter chains at tree depth 7.
        config = replace(
            config,
            search=replace(config.search, n_starts=32),
            sampling=SamplingConfig(warmup=500, samples=500, max_tree_depth=7, target_accept=0.8),
            members=replace(config.members, far_field_cut_radii=2.0, classification_prior_quantile=None, min_deflection_arcsec=0.03),
        )
    return config

def main(argv=None):
    options = _parse_args(sys.argv if argv is None else argv)
    cluster = options["cluster"]
    if cluster not in CLUSTERS:
        print(f"Unknown cluster: {cluster!r}\n{_usage()}")
        return 2
    runtime = build_runtime(cores)
    cpu_plan = resolve_cpu_plan(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    if not options["dry_run"]:
        configure_cpu_runtime(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    config = build_config(cluster, cores=cores, campaign_name=options["campaign_name"], run_label=options["run_label"],
                          image_metadata=options["image_metadata"], runtime=runtime, search_method=options["search_method"],
                          init_strategy=options["init_strategy"], fast=options["fast"]).validate()
    if options["dry_run"]:
        from lenscluster.pipeline import dry_run
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), **dry_run(config)}, indent=2, default=str))
        return 0
    from lenscluster.pipeline import run
    print(f"[config] cluster={cluster} run_name={config.paths.run_name} output_dir={config.paths.output_dir} "
          f"stage={options['stage']} init={config.search.strategy} chains={runtime.chains} search={config.search.method}/{config.search.staged_method} "
          f"n_starts={config.search.n_starts} fast={options['fast']} sigma_obs_arcsec={config.model.image_constraints.sigma_arcsec}")
    start = time.monotonic()
    try:
        run(config, stage=options["stage"])
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
