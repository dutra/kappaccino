"""Read-only ranking/budget analysis of a saved member-classification ensemble."""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lenscluster.member_tiers import (
    MemberTierThresholds, _reference_inputs, _budget_arrays, finite_logh_change,
)
from lenscluster.smoothed_workflow import _load, _reference_snapshot, _stack_snapshots, _active_parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    saved = json.loads((args.artifacts / 'member_tiers.json').read_text())
    budget = json.loads((args.artifacts / 'member_tier_budgets.json').read_text())
    state, _, _, ev = _load(args.artifacts)
    snapshot = jax.jit(lambda p: _reference_snapshot(ev, p, prior_guards=True))
    snapshots = [jax.device_get(snapshot(jnp.asarray(p))) for p in saved['reference_parameters']]
    assert all(bool(s['reference_valid']) for s in snapshots)
    meta = state.smoothed_model
    data = _reference_inputs(**_stack_snapshots(snapshots), parity=_active_parity(ev),
        source_prior_median=meta['source_scatter_prior_median_arcsec'],
        member_prior_median=meta['member_velocity_scatter_prior_median'], prior_guards=True)
    assert data['valid'] and data['template_valid'].all()
    thresholds = MemberTierThresholds(**budget['thresholds'])
    n = data['m']
    curves, rankings, summary = {}, {}, {}
    for axis, displacement, determinant, pos_limit, h_limit, target in (
        ('sampled', data['delta_alpha'], data['delta_d'], thresholds.sampled_deflection, thresholds.sampled_logh, 10),
        ('exact', data['errors'], data['determinant_errors'], thresholds.template_deflection, thresholds.template_logh, 20),
    ):
        pos = displacement / data['source'][:, None, None]
        h = np.asarray(finite_logh_change(data['mean'][:, None], data['variance'][:, None],
            determinant, data['parity'][:, None]))
        scores = np.maximum(pos / pos_limit, h / h_limit).max(axis=(0, 2))
        order = np.array(sorted(range(n), key=lambda i: (-scores[i], str(saved['member_ids'][i]))))
        ids = [saved['member_ids'][i] for i in order]
        assert ids == budget['rankings'][axis]['member_ids']
        np.testing.assert_allclose(scores[order], budget['rankings'][axis]['scores'], rtol=1e-10)
        rankings[axis] = dict(member_ids=ids, scores=scores[order].tolist())
        rows = []
        for k in range(n + 1):
            remaining = order[k:]
            if axis == 'sampled':
                values = dict(individual_deflection=float(pos[:, remaining].max(initial=0)),
                    individual_logh=float(h[:, remaining].max(initial=0)),
                    absorbed_variance=float((pos[:, remaining]**2).sum(axis=1).max(initial=0)))
            else:
                values = dict(template_deflection=float(pos[:, remaining].sum(axis=1).max(initial=0)),
                    template_logh=float(np.asarray(finite_logh_change(data['mean'], data['variance'],
                        determinant[:, remaining].sum(axis=1), data['parity'])).max(initial=0)))
            rows.append(values)
        curves[axis] = rows
        limits = (dict(individual_deflection=pos_limit, individual_logh=h_limit,
                       absorbed_variance=thresholds.absorbed_variance) if axis == 'sampled'
                  else dict(template_deflection=pos_limit, template_logh=h_limit))
        passing = [k for k, row in enumerate(rows) if all(row[key] < limit for key, limit in limits.items())]
        summary[axis] = dict(target=target, current_limits=limits, required_limits=rows[target],
            relaxation_factors={key: rows[target][key]/limit for key, limit in limits.items()},
            first_passing_ranked_prefix=min(passing),
            score_before_cut=float(scores[order[target-1]]), score_after_cut=float(scores[order[target]]),
            drop_ratio=float(scores[order[target-1]]/scores[order[target]]))
        # Cross-check suffix curves against the production aggregate evaluator.
        for k in (0, target, n):
            mask = np.zeros(n, bool)
            mask[order[:k]] = True
            p, h_budget, v = _budget_arrays(data, mask, mask, thresholds)
            if axis == 'sampled':
                np.testing.assert_allclose(rows[k]['absorbed_variance'], v.max()*thresholds.absorbed_variance, rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_allclose([rows[k]['template_deflection'], rows[k]['template_logh']],
                    [p.max()*pos_limit, h_budget.max()*h_limit], rtol=1e-12, atol=1e-12)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope='saved classification references only; fixed impact-ranked prefixes, not globally optimal subsets',
        artifacts=str(args.artifacts.resolve()), reference_count=data['r'], member_count=n,
        prior_guards=True, summary=summary, rankings=rankings,
        reference_parameters=saved['reference_parameters'], reference_parameter_names=saved['reference_parameter_names'])
    (args.output/'tradeoffs.json').write_text(json.dumps(report, indent=2)+'\n')
    for axis, rows in curves.items():
        with (args.output/f'{axis}_budgets.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['count', *rows[0]])
            writer.writeheader()
            writer.writerows(dict(count=k, **row) for k, row in enumerate(rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    labels = dict(individual_deflection='Individual deflection', individual_logh=r'Individual $|\Delta\log h|$',
        absorbed_variance='Total absorbed variance', template_deflection='Summed deflection error',
        template_logh=r'Collective $|\Delta\log h|$')
    for col, axis in enumerate(('sampled', 'exact')):
        target = summary[axis]['target']
        ax = axes[0, col]
        ax.plot(np.arange(1, n+1), rankings[axis]['scores'], color='black')
        ax.set_ylabel('Individual impact / current tolerance')
        ax.set_xlabel(f'Member rank ({axis} selection)')
        ax = axes[1, col]
        for key, limit in summary[axis]['current_limits'].items():
            ax.plot(np.arange(n+1), np.maximum([r[key]/limit for r in curves[axis]], 1e-14), label=labels[key])
        ax.set_ylabel('Remaining error / current tolerance')
        ax.set_xlabel(f'Number treated as {axis}')
        ax.legend(fontsize=9, loc='upper right')
        for ax in axes[:, col]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, n)
            ax.set_ylim(bottom=1e-3)
            ax.axhline(1, color='0.4', linestyle=':', linewidth=1)
            ax.axvline(target, color='firebrick', linestyle='--', linewidth=1)
            ax.text(target*1.07, .04, str(target), color='firebrick', transform=ax.get_xaxis_transform())
            ax.grid(False)
    fig.savefig(args.output/'member_cut_tradeoffs.pdf')
    fig.savefig(args.output/'member_cut_tradeoffs.png', dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
