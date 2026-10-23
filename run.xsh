$XONSH_SHOW_TRACEBACK = True

import json
import os
import sys
import time
from pathlib import Path

cores = 4  # NUTS chains; one JAX CPU device per chain.
seed = 12345
pos_sigma_arcsec = 0.5

$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lenscluster.config import (
    CosmologyConfig,
    DPIEHaloConfig,
    ImageCatalogCutoutConfig,
    ImageConstraintsConfig,
    ImageDiagnosticsConfig,
    IndependentMemberHaloConfig,
    LensClusterSolverConfig,
    LensModelConfig,
    LikelihoodConfig,
    LogNormalPriorConfig,
    MemberConfig,
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
        "Usage: xonsh run.xsh <cluster> [--campaign NAME] [--run-label NAME] [--image-metadata PATH] [--init hybrid|multistart|staged] [--search lbfgs|adam|svi] [--stage init|final|both] [--dry-run]\n"
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
    values = {"campaign": OUTPUT_DIR_LABEL, "run_label": None, "image_metadata": None, "dry_run": False, "search": "lbfgs", "stage": "both", "init": "multistart"}
    flag_to_key = {"--campaign": "campaign", "--run-label": "run_label", "--image-metadata": "image_metadata", "--search": "search", "--stage": "stage", "--init": "init"}
    index = 2
    while index < len(argv):
        flag = argv[index]
        if flag == "--dry-run":
            values["dry_run"] = True
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
                dry_run=values["dry_run"], image_metadata=values["image_metadata"], search_method=values["search"], stage=values["stage"], init_strategy=values["init"])


FF_SIMS_CLUSTERS = {
    "ARES": {
        "cluster_key": "ares",
        "output_dir": f"results/{OUTPUT_DIR_LABEL}/ares",
        "image_catalog": "data/ff_sims/ares/ares_obs_arcs.cat",
        "member_catalog": "data/ff_sims/ares/ares_cluster_members_potfile.cat",
        "cosmology": CosmologyConfig(H0=70.4, Om0=0.272, Ode0=0.728),
        "z_lens": 0.5,
        "halos": (
            DPIEHaloConfig(
                id="1",
                x_centre=20.0,
                y_centre=-32.0,
                ellipticite=0.3,
                angle_pos=0.0,
                core_radius_kpc=20.0,
                cut_radius_kpc=1500.0,
                v_disp=950.0,
                z_lens=0.5,
                priors={
                    "x_centre": PriorConfig("uniform", lower=15.0, upper=25.0, step=0.5),
                    "y_centre": PriorConfig("uniform", lower=-37.0, upper=-27.0, step=0.5),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.5, step=0.02),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=5.0, upper=60.0, step=1.0),
                    "v_disp": PriorConfig("truncated_normal", mean=950.0, std=250.0, lower=500.0, upper=1800.0),
                },
            ),
            DPIEHaloConfig(
                id="2",
                x_centre=-40.0,
                y_centre=40.0,
                ellipticite=0.3,
                angle_pos=0.0,
                core_radius_kpc=20.0,
                cut_radius_kpc=1500.0,
                v_disp=950.0,
                z_lens=0.5,
                priors={
                    "x_centre": PriorConfig("uniform", lower=-45.0, upper=-35.0, step=0.5),
                    "y_centre": PriorConfig("uniform", lower=35.0, upper=45.0, step=0.5),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.5, step=0.02),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=5.0, upper=45.0, step=1.0),
                    "v_disp": PriorConfig("truncated_normal", mean=950.0, std=175.0, lower=600.0, upper=1400.0),
                },
            ),
            ShearHaloConfig(
                id="S1",
                gamma=0.0,
                angle_pos=0.0,
                z_lens=0.5,
                priors={
                    "gamma": PriorConfig("uniform", lower=0.0, upper=0.3, step=0.005),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=0.5),
                },
            ),
        ),
        "member_population": MemberPopulationConfig(
            id="potfile_1",
            catalog_path="data/ff_sims/ares/ares_cluster_members_potfile.cat",
            mag0="auto",
            prior_mag0=18.5,
            corekpc=0.15,
            sigma=100.0,
            cutkpc=270.0,
            z_lens=0.5,
            sigma_prior=PriorConfig("truncated_normal", mean=100.0, std=15.0, lower=70.0, upper=500.0),
            cutkpc_prior=PriorConfig("truncated_normal", mean=270.0, std=35.0, lower=160.0, upper=800.0),
        ),
        "independent_member_halos": (
            IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="2"),
            IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="3"),
        ),
        "kappa_true_fits": "data/ff_sims/published/ares/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/published/ares/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/published/ares/gammay_z9_0.fits",
        "softening_length_kpc": 0.0,
        "softening_length_prior_log_sigma": 0.15,
    },
    "HERA": {
        "cluster_key": "hera",
        "output_dir": f"results/{OUTPUT_DIR_LABEL}/hera",
        "image_catalog": "data/ff_sims/hera/hera_obs_arcs.cat",
        "member_catalog": "data/ff_sims/hera/hera_cluster_members_potfile.cat",
        "cosmology": CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
        "z_lens": 0.507,
        "halos": (
            DPIEHaloConfig(
                id="1",
                x_centre=19.54080009,
                y_centre=2.37820005,
                ellipticite=0.3,
                angle_pos=30.0,
                core_radius_kpc=8.0,
                cut_radius_kpc=1500.0,
                v_disp=800.0,
                z_lens=0.507,
                priors={
                    "x_centre": PriorConfig("uniform", lower=14.54080009, upper=24.54080009, step=0.5),
                    "y_centre": PriorConfig("uniform", lower=-2.62179995, upper=7.37820005, step=0.5),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.8, step=0.02),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=0.5, upper=100.0, step=1.0),
                    # Sampled with an informative prior: cluster-scale cut radii are expected at 1-2 Mpc and the arcs do not constrain them.
                    "cut_radius_kpc": PriorConfig("truncated_normal", mean=1500.0, std=300.0, lower=500.0, upper=3000.0),
                    "v_disp": PriorConfig("truncated_normal", mean=800.0, std=280.0, lower=100.0, upper=2200.0),
                },
            ),
            DPIEHaloConfig(
                id="2",
                x_centre=0.001,
                y_centre=0.003,
                ellipticite=0.3,
                angle_pos=24.0,
                core_radius_kpc=5.0,
                cut_radius_kpc=1500.0,
                v_disp=700.0,
                z_lens=0.507,
                priors={
                    "x_centre": PriorConfig("uniform", lower=-4.999, upper=5.001, step=0.5),
                    "y_centre": PriorConfig("uniform", lower=-4.997, upper=5.003, step=0.5),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.8, step=0.02),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=0.5, upper=100.0, step=1.0),
                    # Sampled with an informative prior: cluster-scale cut radii are expected at 1-2 Mpc and the arcs do not constrain them.
                    "cut_radius_kpc": PriorConfig("truncated_normal", mean=1500.0, std=300.0, lower=500.0, upper=3000.0),
                    "v_disp": PriorConfig("truncated_normal", mean=700.0, std=245.0, lower=100.0, upper=2200.0),
                },
            ),
            ShearHaloConfig(
                id="S1",
                gamma=0.0,
                angle_pos=0.0,
                z_lens=0.507,
                priors={
                    "gamma": PriorConfig("uniform", lower=0.0, upper=0.3, step=0.005),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=0.5),
                },
            ),
        ),
        "member_population": MemberPopulationConfig(
            id="potfile_1",
            catalog_path="data/ff_sims/hera/hera_cluster_members_potfile.cat",
            mag0="auto",
            prior_mag0=19.82,
            corekpc=0.15,
            sigma=200.0,
            cutkpc=12.0,
            z_lens=0.507,
            # Priors at m_F160W = 19.82, about L* at z = 0.507: central dispersion ~220 km/s and r_cut ~10 (+13, -7) kpc from the
            # MUSE-calibrated relations of Bergamini et al. 2019, widened because the mock's galaxies need not share real M/L.
            sigma_prior=PriorConfig("truncated_normal", mean=200.0, std=70.0, lower=40.0, upper=500.0),
            cutkpc_prior=PriorConfig("truncated_normal", mean=12.0, std=10.0, lower=2.0, upper=60.0),
        ),
        # Members 1 and 2 (the two central galaxies) stay in the scaling relation: on Hera the independent halos cost 12
        # parameters for 7 in log-likelihood and made the chains multi-well; the classifier makes them exact and they
        # carry their own deviation.
        "independent_member_halos": (),
        "kappa_true_fits": "data/ff_sims/published/hera/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/published/hera/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/published/hera/gammay_z9_0.fits",
        # HERA uses H0=72 km/s/Mpc, so h=0.72 and 2.3 h^-1 kpc = 3.19 kpc physical.
        "softening_length_kpc": 0.0,
        #"softening_length_kpc": 2.3 / 0.72,
        "softening_length_prior_log_sigma": 0.15,
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
) -> LensClusterSolverConfig:
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be one of {', '.join(sorted(CLUSTERS))}; got {cluster!r}.")
    is_hff = cluster not in FF_SIMS_CLUSTERS
    cluster_config = _hff_clusters()[cluster] if is_hff else FF_SIMS_CLUSTERS[cluster]
    output_dir = f"results/{campaign_name}/{cluster_config['cluster_key']}"
    run_name = run_label if run_label is not None else "smoothed-local-image-plane"

    return LensClusterSolverConfig(
        model=LensModelConfig(
            reference=cluster_config.get("reference", ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0)),
            cosmology=cluster_config["cosmology"],
            large_halos=cluster_config["halos"],
            independent_member_halos=cluster_config["independent_member_halos"],
            member_populations=(cluster_config["member_population"],),
            image_constraints=ImageConstraintsConfig(
                catalog_path=cluster_config["image_catalog"], sigma_arcsec=pos_sigma_arcsec,
                sigma_floor_arcsec=0.5, metadata_path=image_metadata,
            ),
        ),
        paths=RunPathsConfig(output_dir=output_dir, run_name=run_name, figure_prefix=cluster_config["cluster_key"]),
        runtime=runtime if runtime is not None else build_runtime(cores),
        search=SearchConfig(strategy=init_strategy, method=search_method, staged_method=search_method if search_method != "lbfgs" else "adam",
                            n_starts=32, lbfgs_max_iter=400, mode_margin=10.0, max_chain_modes=4, polish_scatter=0.1, search_latent=True,
                            staged_starts=8, staged_steps=500, staged_learning_rate=0.01, svi_particles=1),
        sampling=SamplingConfig(warmup=500, samples=250, max_tree_depth=8, target_accept=0.85),
        members=MemberConfig(far_field_cut_radii=3.0, classification_prior_quantile=0.975, template_tolerance=0.1, top_exact=None),
        scaling=ScalingModelConfig(
            # Members leave the relation in dispersion (tight: Faber-Jackson scatter is about 0.1 dex) and, for the exact
            # members only, in cut radius (loose: tidal truncation depends on the orbit).  With a single dispersion
            # deviation the sampler turned the two biggest subhalos into 700 km/s point masses to patch mass at 20-60".
            member_deviation="sigma_rcut",
            radius_deviation_members="exact",
            member_velocity_scatter_prior=LogNormalPriorConfig(median=0.10, log_sigma=0.3),
            member_radius_scatter_prior=LogNormalPriorConfig(median=0.3, log_sigma=0.5),
            # Faber-Jackson slope 0.27-0.28 and beta_cut 0.64-0.66 in three MUSE-calibrated clusters (Bergamini et al. 2019).
            alpha_sigma_prior_mean=0.27,
            alpha_sigma_prior_std=0.05,
            alpha_sigma_prior_lower=0.05,
            alpha_sigma_prior_upper=0.50,
            beta_radius_prior_mean=0.65,
            beta_radius_prior_std=0.15,
            beta_radius_prior_lower=0.0,
            beta_radius_prior_upper=1.5,
            softening_length_kpc=float(cluster_config["softening_length_kpc"]),
            softening_length_prior_log_sigma=float(cluster_config["softening_length_prior_log_sigma"]),
        ),
        likelihood=LikelihoodConfig(
            source_scatter_prior=LogNormalPriorConfig(median=0.02, log_sigma=0.7),
            source_position_prior_mean_arcsec=(0.0, 0.0),
            source_position_prior_sigma_arcsec=100.0,
            pos_sigma_arcsec=pos_sigma_arcsec,
        ),
        image_diagnostics=ImageDiagnosticsConfig(
            posterior_image_diagnostic_draws=32,
            exact_image_min_distance_arcsec=0.5,
            exact_image_precision_limit=1.0e-2,
            exact_image_num_iter_max=100,
            exact_image_finder="local-lm-adaptive",
            exact_image_lm_max_iter=100,
            exact_image_displacement_tol_arcsec=1.0e-4,
            exact_image_identification_tol_arcsec=1.0e-3,
            match_tolerance_arcsec=5.0,
        ),
        truth=TruthRecoveryConfig(
            kappa_true_fits=cluster_config.get("kappa_true_fits"),
            gammax_true_fits=cluster_config.get("gammax_true_fits"),
            gammay_true_fits=cluster_config.get("gammay_true_fits"),
            posterior_truth_recovery_draws=64,
            truth_grid_size=256,
            caustic_source_redshift=9.0,
        ),
        image_catalog=ImageCatalogCutoutConfig(
            image_dir="data/BUFFALO_Images" if is_hff else "data/ff_sims",
            image_scale="auto",
            bands=tuple(HFF_RGB_BANDS) if is_hff else FF_RGB_BANDS,
            rgb=RGBDisplayConfig(**HFF_RGB_DISPLAY) if is_hff else FF_RGB_DISPLAY,
        ),
    )


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
                          init_strategy=options["init_strategy"]).validate()
    if options["dry_run"]:
        from lenscluster.pipeline import dry_run
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), **dry_run(config)}, indent=2, default=str))
        return 0
    from lenscluster.pipeline import run
    print(f"[config] cluster={cluster} run_name={config.paths.run_name} output_dir={config.paths.output_dir} "
          f"stage={options["stage"]} init={config.search.strategy} chains={runtime.chains} search={config.search.method}/{config.search.staged_method} n_starts={config.search.n_starts} sigma_obs_floor_arcsec={config.model.image_constraints.sigma_floor_arcsec}")
    start = time.monotonic()
    try:
        run(config, stage=options["stage"])
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
