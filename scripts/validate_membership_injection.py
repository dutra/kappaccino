"""Membership injection test on a finished run: does the image likelihood recognize a wrong member?

Each case gives one galaxy a prior membership probability (``--prior``, default 0.5) while every other member stays
certain, and computes its posterior membership P(z = 1 | images) with MemberConfig.membership="marginalize":

interloper
    A galaxy that is not part of the true lens, added to the member catalog near the images (a grid of distances to
    the nearest image and magnitudes).  Its posterior should fall below the prior where the images can tell.
member
    A true catalog member that the classifier doubts.  Its posterior should rise above the prior.

The posterior is the mean over the run's posterior draws of each draw's configuration weights.  The draws were fit
without the interlopers, so this measures what the likelihood can tell at the fitted lens, not a joint refit; the
interloper enters on the scaling relation (deviation offset 0).  Writes ``membership_injection.csv`` and
``membership_injection.pdf`` (posterior membership against the galaxy's largest deflection at an image).

Run it from the repository whose relative catalog paths the run used, for example:

    python scripts/validate_membership_injection.py results/tutorial/hera --output results/membership_injection
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from _covariance_plot_cli import load_run

DISTANCES = (1.0, 2.0, 4.0, 8.0)          # arcsec to the nearest image
MAGNITUDES = (19.0, 20.5, 22.0)           # in the scaling-relation band


def _members_document(config):
    import yaml
    population = config.model.member_populations[0]
    path = Path(population.catalog_path)
    if path.suffix.lower() not in {'.yaml', '.yml'}:
        raise SystemExit(f'{path}: the injection test needs a YAML member catalog')
    return population, yaml.safe_load(path.read_text())


def _interloper_positions(xy, rng):
    """One position per (distance, magnitude) case, its nearest image within 20% of the requested distance."""
    cases = []
    for distance in DISTANCES:
        for magnitude in MAGNITUDES:
            for _ in range(10000):
                anchor = xy[rng.integers(len(xy))]
                angle = rng.uniform(0, 2 * np.pi)
                point = anchor + distance * np.array([np.cos(angle), np.sin(angle)])
                nearest = np.min(np.hypot(*(xy - point).T))
                if 0.8 * distance <= nearest <= 1.25 * distance:
                    break
            cases.append((point, magnitude, nearest))
    return cases


def _doubted_members(state, xy, exact_ids):
    """True exact members spanning magnitude and distance: the nearest-image member in each (magnitude, distance) bin."""
    rows = []
    for position, record in enumerate(state.scaling_component_records):
        member_id = state.members.member_ids[position]
        if member_id not in exact_ids or record.get('role') == 'bcg':
            continue
        nearest = float(np.min(np.hypot(xy[:, 0] - record['x_centre'], xy[:, 1] - record['y_centre'])))
        rows.append((member_id, record['catalog_id'], float(record['catalog_mag']), nearest))
    chosen = []
    for low, high in ((-np.inf, 20.0), (20.0, 21.5), (21.5, np.inf)):
        for near, far in ((0.0, 3.0), (3.0, np.inf)):
            in_bin = [r for r in rows if low <= r[2] < high and near <= r[3] < far]
            if in_bin:
                chosen.append(min(in_bin, key=lambda r: r[3]))
    return chosen


def _case(config, base_state, draws, members_doc, population, exact_ids, uncertain_id, prior, workdir, label, log):
    import jax
    import jax.numpy as jnp
    from kappaccino.likelihood.evaluator import Evaluator
    from kappaccino.io.member_catalog import write_member_catalog
    from kappaccino.core.state import build_state, with_member_classes
    catalog = write_member_catalog(workdir / f'members_{label}.yaml', members_doc['members'], band=members_doc['band'],
                                   kinematics_sources=members_doc.get('kinematics_sources'))
    members = replace(config.members, membership='marginalize', membership_default_probability=1.0, membership_max_enumerated=1,
                      membership_calibration_prior=None)
    case_config = replace(config, members=members,
                          model=replace(config.model, member_populations=(replace(population, catalog_path=str(catalog)),)))
    state = build_state(case_config, drop_member_ids=base_state.dropped_member_ids)
    ids = list(state.members.member_ids)
    centered = set(base_state.members.classification.get('centered_members', []))
    state = with_member_classes(state, np.asarray([i for i, m in enumerate(ids) if m in exact_ids]),
                                centered=np.asarray([m in centered for m in ids]))
    # Map the draws by parameter name; the interloper's own deviation offsets stay at their prior centre.
    old = {spec.name: i for i, spec in enumerate(base_state.parameter_specs)}
    theta = np.tile(np.asarray(state.initial_theta, dtype=float), (len(draws), 1))
    for j, spec in enumerate(state.parameter_specs):
        if spec.name in old:
            theta[:, j] = draws[:, old[spec.name]]
    evaluator = Evaluator(state, case_config.likelihood.likelihood_mode)
    position = ids.index(uncertain_id)
    if evaluator.membership_enumerated_positions.tolist() != [position]:
        raise RuntimeError(f'{uncertain_id}: expected it to be the one enumerated member')
    posterior = np.asarray(jax.lax.map(jax.jit(evaluator.membership_posterior), jnp.asarray(theta)))[:, 0]
    physical = evaluator.physical(jnp.asarray(np.median(theta, axis=0)))
    alpha = np.asarray(evaluator.fields(physical)['member_alpha'])[position] / prior   # at full mass
    deflection = float(np.max(np.hypot(alpha[:, 0], alpha[:, 1])))
    log(f'{label}: {uncertain_id} prior {prior:.2f} -> posterior {posterior.mean():.3f} +- {posterior.std():.3f} '
        f'(largest deflection at an image {deflection:.3f}")')
    return posterior, deflection


def _figure(table, path, prior):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from importlib.resources import files
    from kappaccino.plotting.style import PAPER_STYLE
    plt.style.use(str(files('kappaccino') / 'plotting' / 'style.mplstyle'))
    plt.rcParams.update(PAPER_STYLE)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for kind, color, marker in (('interloper', 'tab:red', 'o'), ('member', 'tab:blue', 's')):
        rows = table[table.kind == kind]
        ax.errorbar(rows.max_deflection_arcsec, rows.p_posterior, yerr=rows.p_posterior_std, fmt=marker, color=color, ms=4,
                    lw=.8, label={'interloper': 'Injected interloper', 'member': 'Doubted true member'}[kind])
    ax.axhline(prior, color='.4', ls='--', lw=.8, label='Prior')
    ax.set_xscale('log')
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('Largest deflection at an image (arcsec)')
    ax.set_ylabel('Posterior membership')
    ax.legend(loc='center right', frameon=False)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_directory')
    parser.add_argument('--output', default='membership_injection', help='output directory')
    parser.add_argument('--prior', type=float, default=0.5, help='prior membership probability of the tested galaxy')
    parser.add_argument('--draws', type=int, default=100, help='posterior draws used (evenly spaced)')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    import pandas as pd
    from kappaccino.io.sky import offsets_to_radec
    output = Path(args.output)
    workdir = output / 'catalogs'
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = (output / 'membership_injection.log').open('w')

    def log(message):
        print(message, flush=True)
        log_file.write(message + '\n')

    config, state, evaluator, posterior, _ = load_run(args.run_directory)
    samples = np.asarray(posterior.samples, dtype=float)
    draws = samples[np.unique(np.linspace(0, len(samples) - 1, min(args.draws, len(samples))).astype(int))]
    population, document = _members_document(config)
    xy = np.asarray(evaluator.xy)
    exact_ids = {state.members.member_ids[i] for i in state.members.exact_positions}
    ra0, dec0 = float(config.model.reference.ra0_deg), float(config.model.reference.dec0_deg)
    prefix = f'{population.id}:'
    rng = np.random.default_rng(args.seed)
    rows = []
    for k, (point, magnitude, nearest) in enumerate(_interloper_positions(xy, rng)):
        name = f'inj{k}'
        ra, dec = offsets_to_radec(point[0], point[1], ra0, dec0)
        doc = json.loads(json.dumps(document))
        doc['members'][name] = {'ra': float(ra), 'dec': float(dec), 'mag': {document['band']: float(magnitude)}, 'p_member': args.prior}
        p, deflection = _case(config, state, draws, doc, population, exact_ids | {prefix + name}, prefix + name, args.prior, workdir,
                              f'interloper_{k}', log)
        rows.append(dict(kind='interloper', member=name, mag=magnitude, nearest_image_arcsec=nearest, max_deflection_arcsec=deflection,
                         p_prior=args.prior, p_posterior=p.mean(), p_posterior_std=p.std()))
    for k, (member_id, catalog_id, magnitude, nearest) in enumerate(_doubted_members(state, xy, exact_ids)):
        doc = json.loads(json.dumps(document))
        doc['members'][catalog_id]['p_member'] = args.prior
        p, deflection = _case(config, state, draws, doc, population, exact_ids, member_id, args.prior, workdir, f'member_{k}', log)
        rows.append(dict(kind='member', member=catalog_id, mag=magnitude, nearest_image_arcsec=nearest, max_deflection_arcsec=deflection,
                         p_prior=args.prior, p_posterior=p.mean(), p_posterior_std=p.std()))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'membership_injection.csv', index=False)
    _figure(table, output / 'membership_injection.pdf', args.prior)
    log(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    log_file.close()


if __name__ == '__main__':
    main()
