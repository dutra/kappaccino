"""Configured paper cutouts rendered directly from saved posterior diagnostics."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .fold_pairs import load_fold_data
from .rgb_overview_cutouts import overview_cutout_data, render_overview_cutouts
from .style import CRITICAL_COLOR


def resolve_panel(selector, systems):
    """Resolve an explicit image, family, or system selector without guessing IDs."""
    kind, separator, identity = selector.partition(':')
    if not separator or kind not in {'image', 'family', 'system'} or not identity:
        raise ValueError(f'Invalid paper cutout {selector!r}; use image:ID, family:ID, or system:ID')
    if kind == 'image':
        return kind, identity, None
    matches = [s for s in systems if (s['system_id'] == identity if kind == 'system'
                                     else identity in s['family_ids'])]
    if len(matches) != 1:
        raise ValueError(f'Paper cutout {selector!r} requires exactly one saved fold region; found {len(matches)}')
    system = dict(matches[0])
    if kind == 'family':
        indices = np.flatnonzero(np.asarray(system['image_family_index']) == system['family_ids'].index(identity))
        system['labels'] = [system['labels'][i] for i in indices]
        system['observed'] = np.asarray(system['observed'])[indices]
        system['positions'] = np.asarray(system['positions'])[:, indices]
        system['image_family_index'] = np.asarray(system['image_family_index'])[indices]
        system['panel_label'] = f'Family {identity}'
        # Keep the parent fold framing and critical fields, selecting only the
        # requested knot's positions and contours.
    return kind, system['system_id'], system


def write_paper_cutouts(config, state, draws, tables, figures, imaging, *, log=print):
    """Create the selected figures after position and fold products are saved.  A figure with a panel that has no
    saved posterior (an image recovered in fewer than three reporting draws) is skipped with a log line."""
    groups = [('small_lambda_posteriors', config.image_catalog.paper_small_lambda_cutouts),
              ('fold_cutouts', config.image_catalog.paper_fold_cutouts)]
    if not any(selectors for _, selectors in groups):
        return []
    tables, figures = Path(tables), Path(figures)
    systems = load_fold_data(tables)['systems']
    outputs = []
    for stem, selectors in groups:
        if not selectors:
            continue
        if len(selectors) > 5:
            raise ValueError(f'{stem} supports at most five panels')
        resolved = [resolve_panel(selector, systems) for selector in selectors]
        data = overview_cutout_data(tables, config.paths.figure_prefix,
            image_labels=tuple(dict.fromkeys(identity for kind, identity, _ in resolved if kind == 'image')),
            system_ids=tuple(dict.fromkeys(identity for kind, identity, _ in resolved if kind != 'image')),
            draws=draws, neighbors=True, log=log)
        if data is None:  # overview_cutout_data logged which panel has no saved posterior
            log(f'[outputs] paper cutouts skipped: {stem}')
            continue
        with plt.style.context(str(Path(__file__).with_name('style.mplstyle'))):
            # One row of square panels over a legend strip (stacked entries under a single panel).
            side, strip = 2.05, .26 if len(selectors) > 1 else .62
            fig = plt.figure(figsize=(side * len(selectors), side + strip))
            try:
                for index, (kind, identity, system) in enumerate(resolved):
                    rect = [index/len(selectors), strip/(side+strip), 1/len(selectors), side/(side+strip)]
                    if kind == 'image':
                        item = next(item for item in data['images'] if item['label'] == identity)
                        panel = dict(images=[item], systems=[])
                    else:
                        panel = dict(images=[], systems=[system])
                    render_overview_cutouts(fig, panel, imaging, state.reference,
                                            image_rects=[rect], system_rects=[rect])
                # Images and contours take the color of their family or knot, so the legend is in a neutral gray.
                color = '0.3'
                fig.legend([Line2D([], [], marker='s', linestyle='none', markerfacecolor='none', markeredgecolor=color,
                                   markersize=4, markeredgewidth=.8),
                            Line2D([], [], color=color, lw=.65),
                            (Patch(facecolor=CRITICAL_COLOR, alpha=.18, edgecolor='none'),
                             Line2D([], [], color=CRITICAL_COLOR, lw=.65))],
                           ['Observed image', 'Predicted observed position', 'Critical curve'],
                           loc='lower center', bbox_to_anchor=(.5, 0.), ncol=3 if len(selectors) > 1 else 1,
                           frameon=False, fontsize=8, borderaxespad=.3, columnspacing=2.)
                output = figures / f'{imaging[3]}_{stem}.pdf'
                fig.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0)
            finally:
                plt.close(fig)
        output.with_suffix('.json').write_text(json.dumps(dict(
            run=config.paths.run_name, selectors=list(selectors), source_tables=str(tables),
            layout='one row over a legend strip', position_probability=.954499736,
            position_contours='predicted observed positions, conditional on successful image recovery',
            family_framing='parent fold region', bands=list(imaging[1])), indent=2)+'\n')
        outputs.append(output)
        log(f'[outputs] paper cutouts written: {output}')
    return outputs
