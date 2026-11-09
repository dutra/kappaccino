from types import SimpleNamespace
import numpy as np
from kappaccino.plotting.mass_figure import shared_grf_convergence


def test_shared_grf_redshift_scaling_and_chunk_padding(monkeypatch):
    from kappaccino.lens import cosmology
    monkeypatch.setattr(cosmology, 'dpie_sigma0_factor_from_config', lambda zl,z,c: z)
    def correction(x,y,f):
        # Family zero has weight 3: alpha_x=3*x^2, alpha_y=3*y^2.
        return 3*x*x,3*y*y,6*x,np.zeros_like(x),np.zeros_like(x),6*y
    e=SimpleNamespace(gp_knee=True,state=SimpleNamespace(gp_blocks='all',z_lens=.3,
        gp_reference_redshift=2.,cosmo_config=None),gp_weights_per_family=np.array([3.]),
        gp_deflection_correction=lambda theta:correction)
    x,y=np.meshgrid(np.arange(3.),np.arange(2.))
    result=shared_grf_convergence(e,None,dict(x=x,y=y),4.,chunk=4)
    np.testing.assert_allclose(result,2*(x+y))


def test_no_shared_map_without_a_mass_field():
    for blocks in ('all','system'):
        e=SimpleNamespace(gp_knee=True,gp_sigma_alpha_index=-1,state=SimpleNamespace(gp_blocks=blocks))
        assert shared_grf_convergence(e,None,None,2.) is None
    assert shared_grf_convergence(SimpleNamespace(gp_knee=False,state=SimpleNamespace(gp_blocks='all')),None,None,2.) is None
