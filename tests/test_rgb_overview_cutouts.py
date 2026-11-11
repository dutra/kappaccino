"""Composite ordering and position contour levels."""
from types import SimpleNamespace as NS
import matplotlib.pyplot as plt
import numpy as np
from kappaccino.plotting import rgb_overview_cutouts as rc
from kappaccino.io.hdf5 import save_arrays
IMAGE_LABELS=('K34.b','K34.a','K33.e','K77.b')
SYSTEM_IDS=('K1','K6','K8','K12')


def test_panel_order_and_only_two_sigma(monkeypatch):
    monkeypatch.setattr(rc,'rgb_reference_band',lambda b:b[0])
    monkeypatch.setattr(rc,'arcsec_to_skycoord',lambda *a:None)
    monkeypatch.setattr(rc,'arcsec_to_radec',lambda x,y,*a:(x,y))
    monkeypatch.setattr(rc,'cutout_window',lambda *a:NS(x0=0,y0=0))
    monkeypatch.setattr(rc,'rgb_cutout',lambda *a:np.zeros((100,100,3)))
    monkeypatch.setattr(rc,'critical_curve_quantiles',lambda *a:[])
    axis=np.linspace(-1,1,40);x,y=np.meshgrid(axis,axis)
    grid=dict(x=x,y=y,density=np.exp(-x*x-y*y),levels=[.3,.7])
    monkeypatch.setattr(rc,'_credible_density_grid',lambda *a:grid)
    data=dict(images=[dict(label=k,center=[0,0],fov=3,z=2,grid=grid,critical=dict(median_determinant=np.ones((4,4)),lower=np.ones((4,4)),upper=np.ones((4,4)))) for k in IMAGE_LABELS],
        systems=[dict(system_id=k,center=[0,0],fov=3,z_source=2,critical=[],seeds=[],origin=[],normal=[],
                      labels=['a'],image_family_index=[0],positions=np.zeros((20,1,2)),observed=[[0,0]]) for k in SYSTEM_IDS])
    ref=NS(pixel_scale_arcsec=.03,wcs=NS(world_to_pixel_values=lambda x,y:(x*40+50,y*40+50)))
    fig=plt.figure()
    info=rc.render_overview_cutouts(fig,data,({'f':ref},('f',),None,'m0416'),(3,0,0))
    assert len(fig.axes)==8
    assert info['top']==list(IMAGE_LABELS) and info['right']==list(SYSTEM_IDS)
    for ax in fig.axes:
        assert len(ax.collections)==1
        np.testing.assert_allclose(ax.collections[0].levels,[.3])
    assert [ax.texts[0].get_text().split('\n')[0] for ax in fig.axes]==list(IMAGE_LABELS)+['System '+k for k in SYSTEM_IDS]
    for ax in fig.axes[4:]:
        marker=next(line for line in ax.lines if line.get_marker()=='s')
        label=next(text for text in ax.texts if text.get_text()=='a')
        assert marker.get_color()==label.get_color()=='#0072B2'
    plt.close(fig)


def test_configured_panel_order_roundtrip_and_limit(tmp_path):
    from dataclasses import asdict
    from kappaccino.config import ImageCatalogCutoutConfig
    import json
    import pytest
    config=ImageCatalogCutoutConfig(paper_small_lambda_cutouts=IMAGE_LABELS,
                                   paper_fold_cutouts=SYSTEM_IDS)
    saved=json.loads(json.dumps(asdict(config)))
    assert saved['paper_small_lambda_cutouts']==list(IMAGE_LABELS)
    assert saved['paper_fold_cutouts']==list(SYSTEM_IDS)
    assert rc.overview_cutout_data(tmp_path,'unused') is None
    with pytest.raises(ValueError,match='at most four'):
        rc.overview_cutout_data(tmp_path,'unused',image_labels=('a','b','c','d','e'))


def test_launcher_declares_ordered_panels():
    from pathlib import Path
    import ast
    source=(Path(__file__).resolve().parents[1]/'run.xsh').read_text()
    for name,expected in [('M0416_SMALL_LAMBDA_POSTERIORS',tuple('image:'+s for s in IMAGE_LABELS)),('M0416_FOLD_CUTOUTS',tuple('system:'+s for s in SYSTEM_IDS))]:
        line=next(line for line in source.splitlines() if line.startswith(name+' ='))
        assert ast.literal_eval(line.split('=',1)[1].strip())==expected
    assert 'paper_small_lambda_cutouts=M0416_SMALL_LAMBDA_POSTERIORS' in source
    assert 'paper_fold_cutouts=M0416_FOLD_CUTOUTS' in source


def test_explicit_panels_compute_missing_critical_fields(tmp_path,monkeypatch):
    import json
    import pandas as pd
    from kappaccino.plotting import critical_fields
    pd.DataFrame(dict(image_label=['a','b'],x_arcsec=[0.,1.],y_arcsec=[0.,0.],z_source=[2.,3.],
                      shown=[False,False])).to_csv(tmp_path/'toy_small_lambda_a_image_posteriors_jwst.csv',index=False)
    save_arrays(tmp_path/'posterior_overview_positions.h5',image_labels=np.array(['a','b']),
             recovered=np.random.default_rng(1).normal(0,.02,(32,2,2)),draw_indices=np.arange(32))
    (tmp_path/'posterior_overview_positions.json').write_text(json.dumps(dict(fingerprint='test')))
    (tmp_path/'fold_pairs_source_caustics.json').write_text('{}')
    (tmp_path/'fold_pairs_source_caustics.h5').touch()
    monkeypatch.setattr(rc,'load_fold_data',lambda _:dict(systems=[]))
    calls=[]
    def compute(*args,**kwargs):
        calls.append(args)
        return None,np.ones((2,4,4)),np.zeros((2,4,4)),np.ones((2,4,4))*2
    monkeypatch.setattr(critical_fields,'median_cutout_determinants',compute)
    data=rc.overview_cutout_data(tmp_path,'toy',image_labels=('b','a'),draws=object())
    assert [item['label'] for item in data['images']]==['b','a']
    assert len(calls)==1 and calls[0][3]==[3.,2.]
    assert all('critical' in item for item in data['images'])
