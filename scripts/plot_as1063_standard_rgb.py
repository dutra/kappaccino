"""Regenerate the standard AS1063 JWST RGB from a saved production posterior.

No sampling or edits to existing repository code. An isolated model-code
snapshot reads the older saved parameterization; current RGB plotting modules
provide magnitude-scaled squares, cyan position contours, the median z=9
critical curve and band, a 5 arcsec scale bar, and the stylesheet legend.

Run with the existing lenscluster/lenstronomy Python interpreter. --check-only
verifies the saved model without loading imaging or calculating plot fields.
"""
import argparse
from dataclasses import replace
import io
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/nov4b_densemass_notfast_parity/as1063_jwst/w2000s1000t8_gamma4_allblocks_shear'
MODEL_REVISION = '920c45bacbcb26b6f6565fb5e6800c1253114ec6'  # saved half-normal white-scatter prior


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN,
                        help='Saved run; default is the latest full JWST production run, not a quick experiment')
    parser.add_argument('--image-dir', type=Path, default=ROOT/'data/imaging/as1063/dja_v7.5')
    parser.add_argument('--images', type=Path, default=ROOT/'catalogs/as1063/diego2026/images.yaml')
    parser.add_argument('--members', type=Path, default=ROOT/'catalogs/as1063/diego2026/members.yaml')
    parser.add_argument('--output', type=Path, help='Default: the run final/figures/<prefix>_jwst_rgb_critical_z9.pdf')
    parser.add_argument('--model-revision', default=MODEL_REVISION,
                        help='Local Git revision matching the saved parameterization; checks must still pass')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--rgb-stretch', type=float, default=20.0,
                        help='RGB saturation scale; lower values reveal fainter galaxies (default: 20)')
    parser.add_argument('--rgb-q', type=float, default=80.0,
                        help='Asinh compression applied to all RGB channels (default: 80)')
    parser.add_argument('--_model-root', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def isolated_worker(args):
    revision = subprocess.check_output(['git','rev-parse',args.model_revision],cwd=ROOT,text=True).strip()
    with tempfile.TemporaryDirectory(prefix='as1063_rgb_') as directory:
        snapshot = Path(directory)
        archive = subprocess.check_output(['git','archive',revision,'src'],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(snapshot,filter='data')
        for module in ('jwst_rgb_figure.py','image_position_contours.py','reporting_fields.py','grf_plotting.py','plot_style.py','style.mplstyle'):
            shutil.copy2(ROOT/'src/lenscluster'/module,snapshot/'src/lenscluster'/module)
        # The current plotting helpers use the manuscript's renamed GRF
        # amplitude index; the saved model snapshot retains its original name.
        for module in ('reporting_fields.py', 'grf_plotting.py'):
            helper = snapshot/'src/lenscluster'/module
            helper.write_text(helper.read_text().replace('sigma_grf_index', 'gp_sigma_alpha_index'))
        # The new YAML reader is used only for marker photometry. Keep the
        # snapshot's original reader for reconstructing the fitted model.
        shutil.copy2(ROOT/'src/lenscluster/image_catalog.py',snapshot/'src/lenscluster/plotting_image_catalog.py')
        renderer = snapshot/'src/lenscluster/jwst_rgb_figure.py'
        renderer.write_text(renderer.read_text().replace('from .image_catalog import load_image_catalog',
                                                         'from .plotting_image_catalog import load_image_catalog'))
        environment = dict(os.environ)
        environment.setdefault('MPLCONFIGDIR',str(snapshot/'matplotlib'))
        environment.setdefault('NUMBA_CACHE_DIR',str(snapshot/'numba'))
        environment['AS1063_RGB_MODEL_REVISION'] = revision
        command = [sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--_model-root',str(snapshot)]
        subprocess.run(command,cwd=ROOT,env=environment,check=True)


def render(args):
    sys.path.insert(0,str(args._model_root/'src'))
    import jax
    jax.config.update('jax_enable_x64',True)
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pandas as pd
    from lenscluster import artifacts
    from lenscluster.config import RGBDisplayConfig
    from lenscluster.state import build_state, with_member_classes
    from lenscluster.evaluator import Evaluator
    from lenscluster.cutouts import cluster_from_config, load_cluster_imaging
    from lenscluster.jwst_rgb_figure import render_rgb_critical

    run = args.run.resolve()
    init, final = run/'init/artifacts', run/'final'
    config = artifacts.load_config(final/'artifacts')
    if cluster_from_config(config.model, config.paths.run_name) != 'as1063':
        raise ValueError('--run must be an AS1063 fit')
    model = replace(config.model,
        image_constraints=replace(config.model.image_constraints,catalog_path=str(args.images.resolve())),
        member_populations=tuple(replace(p,catalog_path=str(args.members.resolve())) for p in config.model.member_populations))
    config = replace(config,model=model,image_catalog=replace(config.image_catalog,
        image_dir=str(args.image_dir.resolve()),image_scale='auto',figure_suffix='_jwst',
        bands=('F115W','F150W','F200W','F444W'),
        rgb=RGBDisplayConfig(q=args.rgb_q, stretch=args.rgb_stretch)))
    classification = json.loads((init/artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config,drop_member_ids=classification.get('dropped_members',()))
    centered = set(classification.get('centered_members',()))
    state = with_member_classes(state,artifacts.load_exact_positions(init),classification,
                                centered=np.array([member in centered for member in state.members.member_ids]))
    state.initial_theta = artifacts.load_start(init)
    saved = json.loads((init/artifacts.STATE_FILE).read_text())
    if [p.name for p in state.parameter_specs] != [p['name'] for p in saved['parameter_specs']]:
        raise ValueError('Rebuilt parameter order differs from the saved run; no plot was written')
    if list(state.members.member_ids) != saved['member_ids']:
        raise ValueError('Rebuilt member IDs differ from the saved run; no plot was written')
    evaluator = Evaluator(state,config.likelihood.likelihood_mode)
    posterior = artifacts.load_posterior(final/'artifacts')
    if not np.isfinite(posterior.log_prob).any():
        raise ValueError('No finite saved posterior probability')
    best = int(np.nanargmax(posterior.log_prob))
    theta = np.asarray(posterior.samples[best])
    np.testing.assert_allclose(evaluator.reported_physical(theta),posterior.physical[best],rtol=1e-10,atol=1e-10)
    print(f'Verified {run}: {len(state.parameter_specs)} parameters; best saved draw {best}',flush=True)
    if args.check_only:
        return
    catalog = pd.read_csv(final/'tables/image_fit_quality.csv',dtype={'image_label':str})
    try:
        extra = pd.read_csv(final/'tables/image_recovery_extra_images.csv',dtype={'family_id':str})
    except pd.errors.EmptyDataError:
        extra = pd.DataFrame()
    output = args.output or final/'figures'/f'{config.paths.figure_prefix}_jwst_rgb_critical_z9.pdf'
    output = output.resolve()
    # FITS section reads repeatedly seek through gzip streams. Unpack each
    # science mosaic once; keep the original files and configuration unchanged.
    from lenscluster.cutouts import find_rgb_band_paths
    image_paths = find_rgb_band_paths(args.image_dir.resolve(),cluster='as1063',
                                     bands=config.image_catalog.bands,image_scale='auto')
    if any(source.name.endswith('.gz') for source in image_paths.values()):
        cache = Path(tempfile.gettempdir())/'lenscluster_as1063_rgb_imaging'
        cache.mkdir(exist_ok=True)
        for source in image_paths.values():
            target = cache/source.name.removesuffix('.gz')
            stamp = target.with_suffix('.source.json')
            identity = dict(path=str(source),size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns)
            if not (target.exists() and stamp.exists() and json.loads(stamp.read_text()) == identity):
                print(f'Caching imaging: {source.name}',flush=True)
                partial = target.with_suffix('.partial')
                opener = gzip.open if source.name.endswith('.gz') else open
                with opener(source,'rb') as src, partial.open('wb') as dst:
                    shutil.copyfileobj(src,dst,length=8*1024*1024)
                partial.replace(target)
                stamp.write_text(json.dumps(identity))
        config = replace(config,image_catalog=replace(config.image_catalog,image_dir=str(cache)))
    imaging = load_cluster_imaging(config.image_catalog,config.model,config.paths.run_name)
    render_rgb_critical(config,state,evaluator,theta,catalog,extra,imaging,output,final/'tables',
                        best_index=best,posterior=posterior,log=lambda message:print(message,flush=True))
    metadata = json.loads(output.with_suffix('.json').read_text())
    metadata.update(regeneration_model_code_commit=os.environ['AS1063_RGB_MODEL_REVISION'],
                    rgb_display=dict(q=args.rgb_q, stretch=args.rgb_stretch),
                    regeneration_script=str(Path(__file__).resolve()),
                    regeneration_checks=['parameter order','member IDs','best-draw physical values at rtol=atol=1e-10'])
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output,flush=True)


if __name__ == '__main__':
    args = parse_args()
    if args._model_root:
        render(args)
    else:
        isolated_worker(args)
