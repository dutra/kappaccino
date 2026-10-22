"""Bounded real-HERA smoke and synchronized evaluator timing.

Run with /home/dutra/.conda/envs/lenstronomy/bin/python.  The default selects the
first two real image families and eight member catalog rows, retaining the
launcher's explicit independent members.  It runs only two SVI updates and a
single NUTS chain with five warm-up and five retained transitions (depth three).
This exercises execution and numerical finiteness; it cannot assess convergence,
coverage, lens recovery, or the accuracy of the per-image scatter approximation.
All generated inputs and reports are written beneath --output-dir, in /tmp by
default.  No production launcher run is started.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Set before importing JAX or executing the launcher's imports.
os.environ["JAX_NUM_CPU_DEVICES"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lenscluster")


def launcher_config():
    path = ROOT / "run.xsh"
    lines = [line for line in path.read_text().splitlines()
             if not line.lstrip().startswith("$")
             and 'os.environ["JAX_NUM_CPU_DEVICES"]' not in line]
    namespace = {"__file__": str(path), "__name__": "smoothed_validation_launcher"}
    exec(compile("\n".join(lines), str(path), "exec"), namespace)
    return namespace["build_config"]("HERA", cores=1, campaign_name="smoothed_smoke")


def catalog_rows(path):
    lines = Path(path).read_text().splitlines()
    return ([line for line in lines if line.lstrip().startswith("#")],
            [line for line in lines if line.strip() and not line.lstrip().startswith("#")])


def subset_inputs(config, output_dir, family_count, member_count):
    source_images = ROOT / config.model.image_constraints.catalog_path
    headers, rows = catalog_rows(source_images)
    families = list(dict.fromkeys(row.split()[0].split(".")[0] for row in rows))[:family_count]
    selected_images = [row for row in rows if row.split()[0].split(".")[0] in families]
    image_path = output_dir / "hera_images_subset.cat"
    image_path.write_text("\n".join(headers+selected_images)+"\n")
    populations = []
    kept_ids = []
    for population in config.model.member_populations:
        source_members = ROOT / population.catalog_path
        headers, rows = catalog_rows(source_members)
        independent_ids = {member.catalog_id for member in config.model.independent_member_halos
                           if member.population_id == population.id}
        keep = {row.split()[0] for row in rows[:member_count]} | independent_ids
        selected_members = [row for row in rows if row.split()[0] in keep]
        path = output_dir / f"{population.id}_members_subset.cat"
        path.write_text("\n".join(headers+selected_members)+"\n")
        populations.append(replace(population, catalog_path=str(path)))
        kept_ids.extend(row.split()[0] for row in selected_members)
    return replace(config, model=replace(config.model,
        member_populations=tuple(populations),
        image_constraints=replace(config.model.image_constraints, catalog_path=str(image_path), metadata_path=None))), families, kept_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/smoothed_real_hera_smoke"))
    parser.add_argument("--families", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--members", type=int, choices=range(2, 17), default=8)
    parser.add_argument("--benchmark-only", action="store_true")
    cli = parser.parse_args()
    os.chdir(ROOT)
    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report = dict(status="started", purpose="execution_smoke_only", scientific_validation=False,
                  production_inference=False, launcher=str(ROOT/"run.xsh"),
                  launcher_sha256=hashlib.sha256((ROOT/"run.xsh").read_bytes()).hexdigest(),
                  svi_steps=2, nuts_warmup=5, nuts_samples=5, chains=1, max_tree_depth=3,
                  benchmark_target="Evaluator factor: smoothed likelihood plus source transport/prior correction",
                  benchmark_device_synchronization="block_until_ready before and after each timed batch; also after each call",
                  limitations=["Reduced real catalog", "No convergence or coverage claim",
                               "Fixed conservative all-exact/all-sampled tiers for NUTS",
                               "Initializer benchmark uses all members absorbed"])

    import jax
    import jax.numpy as jnp
    import numpy as np
    from lenscluster import cluster_solver as cs
    from lenscluster.config import TruthRecoveryConfig
    from lenscluster.member_tiers import MemberTierRecord
    from lenscluster.planning import compile_run_plan

    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (np.integer, np.bool_)):
            return value.item()
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        return value

    def save():
        report_path.write_text(json.dumps(plain(report), indent=2, allow_nan=False)+"\n")

    def benchmark(name, function, theta):
        start = time.perf_counter()
        compiled = jax.jit(function).lower(theta).compile()
        compilation = time.perf_counter()-start
        for _ in range(3):
            result = jax.block_until_ready(compiled(theta))
        times = []
        for _ in range(3):
            jax.block_until_ready(theta)
            start = time.perf_counter()
            for _ in range(20):
                result = jax.block_until_ready(compiled(theta))
            times.append((time.perf_counter()-start)/20)
        finite = all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
        report.setdefault("benchmarks", {})[name] = dict(compile_seconds=compilation,
            seconds_per_call_median=float(np.median(times)), seconds_per_call_batches=times,
            warmup_calls=3, batches=3, calls_per_batch=20, synchronized=True, finite=finite)
        save()
        print(name, report["benchmarks"][name], flush=True)
        if not finite:
            raise RuntimeError(f"Nonfinite benchmark output: {name}")

    started = time.perf_counter()
    try:
        config, families, member_ids = subset_inputs(launcher_config(), output_dir, cli.families, cli.members)
        config = replace(config,
            paths=replace(config.paths, output_dir=str(output_dir), run_name="reduced_hera_smoke"),
            runtime=replace(config.runtime, chains=1, nuts_chain_method="sequential",
                quiet=True, skip_plots=True, skip_validation=True, numpyro_print_summary=False,
                debug_sampler_diagnostics=False, dense_mass="lens-and-hyperparameters"),
            workflow=replace(config.workflow, stage2_fresh_process=False, exact_image_diagnostics_stage2=False),
            schedule=replace(config.schedule, svi_steps=(2, 2, 2), warmup=(5, 5), samples=(5, 5),
                max_tree_depth=(3, 3), refresh_every=(None, None, None),
                sampling_refresh_runs=(1, 1)), truth=TruthRecoveryConfig())
        (output_dir/"config.json").write_text(json.dumps(config.to_json_dict(), indent=2)+"\n")
        plan = compile_run_plan(config)
        args = cs._clone_args(plan.runtime_args, sample_likelihood_mode="smoothed-local-image-plane",
            sampling_engine="tiered_members", member_tier_phase="initializer", member_tiers_payload=None,
            perturbation_discovery_stage0=False, fit_cosmology_flat_wcdm=False,
            svi_steps=2, warmup=5, samples=5, max_tree_depth=3, chains=1,
            refresh_every=None, quiet=True, nuts_chain_method="sequential", numpyro_print_summary=False)
        state = cs._build_state_from_inputs(args, fit_mode_override="joint")
        evaluator = cs._build_cluster_evaluator_from_args(args, state)
        theta = jnp.asarray(cs._reference_theta_from_init_values(state.parameter_specs, None,
                            cs._default_theta(state.parameter_specs)))
        report.update(jax=jax.__version__, devices=str(jax.devices()), dtype=str(theta.dtype),
            cpu_affinity=sorted(os.sched_getaffinity(0)), selected_family_ids=families,
            selected_member_ids=member_ids, images=len(evaluator.flat_critical_arc_data.x_obs),
            families=len(state.family_data), components=len(state.base_components),
            independent_members=len(config.model.independent_member_halos),
            scaling_members=len(state.smoothed_model["member_ids"]), initializer_parameters=len(theta))
        save()
        benchmark("initializer_likelihood", evaluator._source_loglike_impl, theta)
        benchmark("initializer_value_and_gradient", jax.value_and_grad(evaluator._source_loglike_impl), theta)
        meta = state.smoothed_model
        tiers = MemberTierRecord(tuple(meta["member_ids"]), tuple(meta["member_components"]),
            exact_sampled=tuple(range(len(meta["member_ids"]))), selection_reason="smoke_conservative_without_reference_ensemble")
        args = cs._clone_args(args, member_tier_phase="fixed", member_tiers_payload=tiers.to_dict())
        state = cs._build_state_from_inputs(args, fit_mode_override="joint")
        evaluator = cs._build_cluster_evaluator_from_args(args, state)
        theta = jnp.asarray(cs._reference_theta_from_init_values(state.parameter_specs, None,
                            cs._default_theta(state.parameter_specs)))
        report.update(fixed_parameters=len(theta), fixed_tiers=tiers.to_dict())
        benchmark("fixed_likelihood", evaluator._source_loglike_impl, theta)
        benchmark("fixed_value_and_gradient", jax.value_and_grad(evaluator._source_loglike_impl), theta)
        if not cli.benchmark_only:
            sample_model = cs._posterior_model(state.parameter_specs, evaluator)
            initial_values = {spec.sample_name: float(theta[i]) for i, spec in enumerate(state.parameter_specs)}
            print("starting 2-step SVI", flush=True)
            phase = time.perf_counter()
            seeds, diagnostics, center = cs._run_svi_initializer(args, sample_model, state.parameter_specs,
                jax.random.PRNGKey(12345), init_values=initial_values)
            report["svi"] = dict(seconds=time.perf_counter()-phase, diagnostics=diagnostics,
                                 finite_center=bool(np.isfinite(center).all()))
            save()
            nuts_init = cs._nuts_initialization_from_svi_center(args, state.parameter_specs, center,
                diagnostics, chain_seeds=seeds)
            print("starting 5-warmup/5-draw NUTS", flush=True)
            phase = time.perf_counter()
            posterior = cs._run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init)
            jax.block_until_ready(posterior.samples)
            report["nuts"] = dict(seconds=time.perf_counter()-phase,
                draws=int(len(posterior.samples)), finite_samples=bool(np.isfinite(posterior.samples).all()),
                finite_log_prob=bool(np.isfinite(posterior.log_prob).all()),
                divergent_transitions=int(np.asarray(posterior.diverging).sum()),
                mean_acceptance=float(np.asarray(posterior.accept_prob).mean()),
                leapfrog_steps=np.asarray(posterior.num_steps),
                log_prob_coordinates=posterior.init_diagnostics.get("log_prob_coordinates"),
                diagnostics=posterior.init_diagnostics)
            np.savez_compressed(output_dir/"smoke_posterior.npz", sampling_parameters=posterior.samples,
                log_prob=posterior.log_prob, diverging=posterior.diverging, accept_prob=posterior.accept_prob,
                parameter_names=np.asarray([s.sample_name for s in state.parameter_specs]),
                physical_parameters=np.stack([evaluator.reported_physical_parameter_vector(t) for t in posterior.samples]))
        report.update(status="completed", elapsed_seconds=time.perf_counter()-started)
        save()
        print(str(report_path), flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.perf_counter()-started)
        save()
        raise


if __name__ == "__main__":
    main()
