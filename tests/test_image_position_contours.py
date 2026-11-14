from types import SimpleNamespace
import numpy as np
import pandas as pd
import jax.numpy as jnp
from kappaccino.diagnostics import position_contours as ipc
from kappaccino.core.model import FamilyData
from kappaccino.diagnostics.exact_images import ExactImageSolver
from kappaccino.io.hdf5 import load_arrays, save_arrays


def test_contour_positions_are_predicted_observed_for_catalog_images_and_true_for_extras(tmp_path):
    rows=pd.DataFrame(dict(draw_index=[0,0,1,1],posterior_index=[5,5,7,7],family_id=['a']*4,image_label=['a1','a2']*2,
                           recovered=[True,False,True,True],x_model_arcsec=[5.,9.,5.,1.],y_model_arcsec=[5.,9.,5.,0.],
                           x_predicted_arcsec=[0.,9.,.1,1.],y_predicted_arcsec=[0.,9.,0.,0.],
                           x_true_arcsec=[2.,9.,2.1,1.],y_true_arcsec=[2.,9.,2.,0.]))
    rows.to_csv(tmp_path/'posterior_image_residual_draws.csv',index=False)
    pd.DataFrame(dict(family_id=['a'],x_model_arcsec=[10.],y_model_arcsec=[0.])).to_csv(tmp_path/'image_recovery_extra_images.csv',index=False)
    pd.DataFrame(dict(draw_index=[0,1,1],posterior_index=[5,7,7],family_id=['a','a','a'],x_model_arcsec=[10.2,9.9,30.],y_model_arcsec=[0.,0.,0.],
                      x_predicted_arcsec=[10.5,9.6,30.],y_predicted_arcsec=[.3,-.2,0.],
                      x_true_arcsec=[10.3,9.8,30.],y_true_arcsec=[.1,-.1,0.])).to_csv(
        tmp_path/'posterior_extra_images.csv',index=False)
    data=ipc.position_draws(np.array([5,7]),tmp_path)
    np.testing.assert_allclose(data['recovered'][:,0],[[0.,0.],[.1,0.]])
    assert np.isnan(data['recovered'][0,1]).all()
    np.testing.assert_allclose(data['extra'][:,0],[[10.3,.1],[9.8,-.1]])   # matched by lens position, drawn at the true one
    np.testing.assert_allclose(data['extra_predicted'][:,0],[[10.5,.3],[9.6,-.2]])
    np.testing.assert_array_equal(data['recovered_predicted'],data['recovered'])
    np.testing.assert_allclose(data['recovered_lens'][:,0],[[5.,5.],[5.,5.]])   # the lens positions of the same draws
    assert np.isnan(data['recovered_lens'][0,1]).all()
    np.testing.assert_allclose(data['extra_lens'][:,0],[[10.2,0.],[9.9,0.]])
    np.testing.assert_allclose(data['recovered_true'][:,0],[[2.,2.],[2.1,2.]])
    np.testing.assert_allclose(data['extra_true'][:,0],[[10.3,.1],[9.8,-.1]])
    np.testing.assert_array_equal(data['draw_indices'],[5,7])


def test_gaussian_contour_probabilities_and_no_covariance_fallback():
    rng=np.random.default_rng(83);points=rng.normal(size=(4000,2))*[.1,.2]
    contours,reason=ipc.position_contours(points)
    assert reason is None
    assert [c['probability'] for c in contours]==[.682689492,.954499736]
    # Gaussian KDE broadens slightly; radii should bracket the expected 2D values.
    radii=[np.median(np.linalg.norm(np.concatenate(c['segments'])/[.1,.2],axis=1)) for c in contours]
    np.testing.assert_allclose(radii,np.sqrt(-2*np.log(1-np.array([.682689492,.954499736]))),rtol=.12)
    assert ipc.position_contours(np.zeros((100,2)))[0]==[]
    assert ipc.position_contours(points[:2])[0]==[]
    # Few draws: Gaussian ellipses at the chi-square radii of the sample covariance.
    few,reason=ipc.position_contours(points[:8])
    assert reason is None and [c['probability'] for c in few]==[.682689492,.954499736]


def test_overview_contours_have_tiny_catalog_crosses(monkeypatch):
    import matplotlib.pyplot as plt
    from kappaccino.plotting import mass_figure as mf
    monkeypatch.setattr(mf.cutouts,'rgb_cutout',lambda *a:np.ones((16,16,3)))
    monkeypatch.setattr(mf,'aligned_rgb',lambda rgb,*a:rgb)
    geometry=dict(resolution=16,xlim=(-10.,10.),ylim=(-10.,10.),center_xy=(0.,0.),center=None,size=20.)
    catalog=pd.DataFrame(dict(x_obs_arcsec=[0.,1.],y_obs_arcsec=[0.,1.],x_model_arcsec=[.1,np.nan],y_model_arcsec=[.1,np.nan]))
    contour=dict(kind='recovered',probability=.682689492,segments=[np.array([[0.,0.],[.1,.2],[.3,0.]])])
    fig,ax=plt.subplots()
    mf.overview_axes(fig,ax,None,None,None,None,catalog,pd.DataFrame(),geometry,position_contours={'curves':[contour]})
    assert len(ax.collections)==2
    for collection in ax.collections:
        assert len(collection.get_paths()[0].vertices)==4
        np.testing.assert_array_equal(collection.get_sizes(),[4])
    assert any(line.get_color()=='tab:blue' and line.get_linestyle()=='-' for line in ax.lines)
    labels=[t.get_text() for t in ax.get_legend().get_texts()]
    assert 'Covariance ellipse' not in labels
    assert 'Catalog: missed' in labels
    plt.close(fig)




def test_coverage_of_catalog_positions_is_calibrated_for_draws_of_the_same_distribution():
    rng = np.random.default_rng(11)
    shape = np.array([[.2, .0], [.15, .05]])
    levels = np.array([ipc.credible_level(rng.normal(size=(64, 2)) @ shape, rng.normal(size=2) @ shape) for _ in range(1500)])
    np.testing.assert_allclose([np.mean(levels <= p) for p in ipc.PROBABILITIES], ipc.PROBABILITIES, atol=.04)
    few = np.array([ipc.credible_level(rng.normal(size=(8, 2)), np.zeros(2)) for _ in range(20)])
    assert np.all((few >= 0) & (few <= 1)) and np.median(few) < .2          # few draws: the Gaussian value; the centre is inside
    assert np.isnan(ipc.credible_level(np.zeros((30, 2)), np.zeros(2))) and np.isnan(ipc.credible_level(np.ones((2, 2)), np.zeros(2)))
    draws = rng.normal(size=(40, 2)) * .1
    rows = pd.DataFrame(dict(image_label=['a'] * 40 + ['b'] * 40, family_id='f', recovered=[True] * 79 + [False],
                             x_obs_arcsec=[0.] * 40 + [5.] * 40, y_obs_arcsec=0.,
                             x_predicted_arcsec=np.r_[draws[:, 0], draws[:, 0]], y_predicted_arcsec=np.r_[draws[:, 1], draws[:, 1]]))
    table = ipc.position_coverage(rows)
    assert table.image_label.tolist() == ['a', 'b'] and table.valid_draws.tolist() == [40, 39]
    assert table.inside_68.tolist() == [True, False] and table.inside_95.tolist() == [True, False]       # b's catalog is 5" away
