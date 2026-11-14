"""The shared field behind the reported products: which evaluator supplies it and how its draws are built."""
from types import SimpleNamespace
import numpy as np


def evaluator(monkeypatch):
    from kappaccino.lens import cosmology
    monkeypatch.setattr(cosmology, 'dpie_sigma0_factor_from_config', lambda zl,z,c: z)
    e=SimpleNamespace(gp_knee=True, sigma_grf_index=0, gp_knee_spectrum='knee', state=SimpleNamespace(gp_blocks='all',z_lens=.3,
        gp_reference_redshift=2.,cosmo_config=None), gp_weights_per_family=np.array([3.]))
    e._padded_evaluator=lambda:e
    return e


def test_each_draw_gets_one_realization_of_the_shared_twin_field(monkeypatch):
    """A system-block run reports a field of its all-block twin, with the draw re-expressed for it; the realization of
    a draw is drawn once, from a generator fixed by the run seed and the posterior row, however far it is evaluated."""
    from kappaccino.likelihood import grf_fields
    from kappaccino.plotting import grf
    from kappaccino.reporting import draws as reporting
    twin=evaluator(monkeypatch)
    run=evaluator(monkeypatch)
    run.state.gp_blocks='system'
    run._shared_field_twin=twin
    assert grf_fields.shared_field_evaluator(run) is twin
    monkeypatch.setattr(reporting,'shared_theta',lambda e,theta:('twin',e,tuple(theta)))
    drawn,built=[],[]
    def field_draw(base,theta,rng):
        drawn.append((base,theta,rng.integers(1<<30)))
        return dict(id=len(drawn))
    monkeypatch.setattr(grf,'field_draw',field_draw)
    monkeypatch.setattr(grf,'field_realization',lambda base,draw,reach:built.append((base,draw['id'],reach)) or ('field',draw['id'],reach))
    posterior=SimpleNamespace(samples=np.arange(8.).reshape(4,2),physical=np.arange(8.).reshape(4,2))
    draws=reporting.ReportingDraws(run,posterior,4,7)
    assert draws.correction(2,5.)==('field',1,5.)
    assert draws.correction(2,3.)==('field',1,5.) and len(built)==1          # a shorter reach reuses the field
    assert draws.correction(2,9.)==('field',1,9.) and len(drawn)==1          # a longer reach rebuilds it from the same draw
    assert drawn[0][0] is twin and drawn[0][1]==('twin',run,tuple(draws.samples[2]))
    assert drawn[0][2]==np.random.default_rng([7,int(draws.indices[2])]).integers(1<<30)
    assert draws.correction(0,5.)==('field',2,5.) and drawn[1][2]!=drawn[0][2]
    twin.sigma_grf_index=run.sigma_grf_index=-1                              # no mass field: nothing to add
    assert reporting.ReportingDraws(run,posterior,4,7).correction(0,5.) is None


def test_observed_stretch_uses_the_jacobian_with_the_field():
    import pandas as pd
    from kappaccino.plotting.mass_figure import arc_direction_table
    # Parametric stretch is x, but the conditional mass field changes it to y.
    image=SimpleNamespace(image_labels=['a'],x_obs=np.array([0.]),y_obs=np.array([0.]),
                          family_index_per_image=np.array([0]),family_ids=('f',))
    class Draws:
        evaluator=SimpleNamespace(state=SimpleNamespace(images=image,family_data=[SimpleNamespace(family_id='f',z_source=2.)]))
        def __len__(self):return 2
        def jacobians(self,k,xy,z):
            parametric=np.eye(2)-np.diag([.1,.8])
            return parametric[None],np.diag([-.6,.7])[None]
    result=arc_direction_table(Draws(),pd.DataFrame({'image_label':['a']}))
    np.testing.assert_allclose(result[['e_arc_x','e_arc_y']],[[0.,1.]])
    np.testing.assert_allclose(result.s_arc,.1)


def test_latent_from_physical_reconstructs_gp_sources(monkeypatch):
    import jax.numpy as jnp
    from kappaccino.core import model
    from kappaccino.likelihood.grf_fields import latent_from_physical
    monkeypatch.setattr(model, 'convert_theta_to_latent', lambda p,s:np.array(p,copy=True))
    e=SimpleNamespace(gp_knee=True,state=SimpleNamespace(parameter_specs=[],images=SimpleNamespace(
        source_x_param_index=np.array([1]),source_y_param_index=np.array([2]))),
        gp=dict(knot_mask=np.array([[True]]),knot_family=np.array([[0]]),
                real_block=np.array([0]),real_knot=np.array([0]),real_family=np.array([0])))
    e._padded_evaluator=lambda:e
    e.statistics=lambda t:{'whitening':{'cholesky':jnp.array([[[2.,0.],[0.,2.]]]),
                                      'centroid':jnp.array([[t[0],t[0]]])}}
    e.reported_physical=lambda t:jnp.array([t[0],2*t[1]+t[0],2*t[2]+t[0]])
    p=SimpleNamespace(physical=np.array([[1.,10.,14.],[5.,12.,17.],[9.,20.,18.]]))
    theta=latent_from_physical(e,np.median(p.physical,axis=0))
    np.testing.assert_allclose(theta,[5.,3.5,6.])
    np.testing.assert_allclose(e.reported_physical(theta),np.median(p.physical,axis=0))

