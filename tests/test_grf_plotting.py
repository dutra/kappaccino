from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
from kappaccino.likelihood import gp_knee
from kappaccino.plotting.grf import SmoothPlotDeflection


def toy_field(monkeypatch,modes=()):
    r=np.linspace(0,12,1201);ct=np.exp(-r*r/8);cl=(1-r*r/4)*ct
    monkeypatch.setattr(gp_knee,'knee_curves',lambda *args:(jnp.asarray(cl),jnp.asarray(ct)))
    xy=np.array([[[0.,0.],[1.,.5],[-.7,1.1]]])
    layout=dict(xy=jnp.asarray(xy),mask=jnp.ones((1,3),bool),weights=jnp.ones((1,3)),block_of_family=jnp.array([0,0]))
    tables=dict(r_grid=jnp.asarray(r))
    coef=np.array([[[.2,-.1],[-.3,.4],[.1,.3]]])
    args=(layout,tables,coef,1.,.7,np.array([1.,.6]))
    return SmoothPlotDeflection(*args,modes=modes),args


def test_plot_field_matches_saved_radial_nodes(monkeypatch):
    smooth,args=toy_field(monkeypatch)
    # A one-image layout isolates exact preservation of both radial kernel values.
    layout,tables,coef,lam,amp,weights=args
    layout=dict(xy=jnp.zeros((1,1,2)),mask=jnp.ones((1,1),bool),weights=jnp.ones((1,1)),block_of_family=jnp.array([0,0]))
    coef=coef[:,:1]
    smooth=SmoothPlotDeflection(layout,tables,coef,lam,amp,weights)
    original=gp_knee.MeanDeflectionKnee(layout,tables,coef,lam,amp,weights)
    x=np.linspace(.1,3.,30);y=np.zeros_like(x);families=np.zeros(len(x),int)
    np.testing.assert_allclose(np.array(smooth(x,y,families)[:2]),np.array(original(x,y,families)[:2]),atol=1e-13)


def test_plot_derivatives_are_consistent_continuous_and_curl_free(monkeypatch):
    field,_=toy_field(monkeypatch,modes=('translation','shear'))
    points=np.array([[.04,.08],[.2,.1],[0.,0.],[1.,.5],[-.3,1.]])
    families=np.array([0,1,0,1,0]);c=field(*points.T,families)
    jac=np.stack(c[2:],axis=-1).reshape(-1,2,2)
    numerical=np.empty_like(jac)
    for axis in range(2):
        step=np.zeros(2);step[axis]=1e-6
        a=np.column_stack(field(*(points+step).T,families)[:2])
        b=np.column_stack(field(*(points-step).T,families)[:2])
        numerical[:,:,axis]=(a-b)/2e-6
    np.testing.assert_allclose(jac,numerical,atol=2e-9,rtol=1e-7)
    np.testing.assert_allclose(jac[:,0,1],jac[:,1,0],atol=1e-13)
    jumps=[]
    for eps in [1e-5,1e-6]:
        c=field(np.array([.5-eps,.5+eps]),np.zeros(2),np.zeros(2,int))
        jumps.append(np.linalg.norm(np.array(c[2:])[:,1]-np.array(c[2:])[:,0]))
    assert jumps[1]<.11*jumps[0]


def test_radial_refinement_converges_to_analytic_potential_kernel(monkeypatch):
    layout=dict(xy=jnp.zeros((1,1,2)),mask=jnp.ones((1,1),bool),weights=jnp.ones((1,1)),block_of_family=jnp.array([0]))
    u=np.array([.2,-.1]);points=np.array([[.137,.231],[.793,-.419],[1.233,.831],[2.371,-.325]])
    r2=np.sum(points**2,axis=1);ct=np.exp(-r2/8);du=points@u
    truth=ct[:,None,None]*(
        -(u[None,:,None]*points[:,None,:]+points[:,:,None]*u[None,None,:]+du[:,None,None]*np.eye(2))/4
        +du[:,None,None]*points[:,:,None]*points[:,None,:]/16)
    errors=[]
    for step in [.04,.02,.01]:
        r=np.arange(0.,12.+step,step);ct_nodes=np.exp(-r*r/8);cl_nodes=(1-r*r/4)*ct_nodes
        monkeypatch.setattr(gp_knee,'knee_curves',lambda *args:(jnp.asarray(cl_nodes),jnp.asarray(ct_nodes)))
        field=SmoothPlotDeflection(layout,dict(r_grid=jnp.asarray(r)),u[None,None],1.,1.,np.ones(1))
        actual=np.stack(field(*points.T,np.zeros(len(points),int))[2:],axis=-1).reshape(-1,2,2)
        errors.append(np.max(abs(actual-truth)))
    assert np.all(np.diff(errors)<0)
    assert errors[-1]<1e-6


def test_wide_plot_extends_tables_without_changing_conditioning():
    from kappaccino.plotting.grf import display_tables
    tables=gp_knee.knee_tables(r_max=2.,r_step=.05,lam_min=1.,lam_max=4.,n_lam=4,
        scale_min=.5,scale_max=10.,gamma=4.)
    base=SimpleNamespace(gp_tables=tables)
    assert display_tables(base) is tables and display_tables(base,1.5) is tables
    extended=display_tables(base,4.)
    assert base.gp_tables is tables and float(tables['r_grid'][-1])==2. and float(extended['r_grid'][-1])>=4.
    for key in ('c_l','c_t','d2_l','d2_t'):
        np.testing.assert_array_equal(extended[key][:len(tables['r_grid'])],tables[key])
    assert display_tables(base,3.) is extended


def _toy_conditioning(modes,seed=0):
    """Five images of one block with unequal weights, data covariance C = W K_r W + noise^2, and some data."""
    from kappaccino.plotting.grf import image_field_covariance
    rng=np.random.default_rng(seed)
    tables=gp_knee.knee_tables(r_max=12.,r_step=.05,lam_min=1.,lam_max=8.,n_lam=4,scale_min=.5,scale_max=40.,gamma=4.)
    xy=rng.uniform(-3,3,(5,2));weights=np.array([1.,.8,1.2,1.,.9]);sigma,lam,noise=.3,4.,.05
    layout=dict(xy=jnp.asarray(xy[None]),mask=jnp.ones((1,5),bool),weights=jnp.asarray(weights[None]),block_of_family=jnp.array([0]))
    cl,ct=map(np.asarray,gp_knee.knee_curves(tables,jnp.asarray(lam),None))
    covariance=lambda points:image_field_covariance(points,np.ones(len(points),bool),np.asarray(tables['r_grid']),cl,ct,sigma**2
                                                    ).transpose(0,2,1,3).reshape(2*len(points),2*len(points))
    k=covariance(xy)
    if modes:
        q=np.asarray(gp_knee.low_order_projector(layout['xy'],layout['mask'],modes)[0])[0]
        projector=np.eye(10)-q@q.T
        k=projector@k@projector
    w2=np.repeat(weights,2)
    c=k*w2[:,None]*w2[None]+noise**2*np.eye(10)
    cholesky=np.linalg.cholesky(c)
    data=cholesky@rng.normal(size=10)
    v=np.linalg.solve(c,data).reshape(1,5,2)
    return dict(layout=layout,tables=tables,xy=xy,w2=w2,k=k,c=c,cholesky=cholesky[None],v=v,sigma=sigma,lam=lam,covariance=covariance)


def _realizations(toy,points,modes,count):
    from kappaccino.plotting.grf import FieldRealization,draw_field
    rng=np.random.default_rng(11)
    fields=[]
    for _ in range(count):
        draw=draw_field(toy['layout'],toy['tables'],toy['cholesky'],toy['v'],toy['sigma'],toy['lam'],4.,modes=modes,rng=rng,n_modes=256)
        fields.append(np.column_stack(FieldRealization(toy['layout'],toy['tables'],draw,np.ones(1),modes=modes)(*points.T,np.zeros(len(points),int))[:2]).ravel())
    return np.asarray(fields)


def test_realizations_have_the_conditional_mean_and_covariance_away_from_the_images():
    """Matheron's rule: the sample mean is the conditional mean M[v] and the sample covariance is K** - K* W C^-1 W K*^T."""
    toy=_toy_conditioning(())
    points=np.array([[.5,.2],[2.5,-1.],[6.,5.]])
    fields=_realizations(toy,points,(),1200)
    joint=toy['covariance'](np.vstack([points,toy['xy']]))
    cross=joint[:6,6:]*toy['w2'][None]
    expected_covariance=joint[:6,:6]-cross@np.linalg.solve(toy['c'],cross.T)
    expected_mean=np.column_stack(SmoothPlotDeflection(toy['layout'],toy['tables'],toy['v'],toy['lam'],toy['sigma']**2,np.ones(1))(
        *points.T,np.zeros(3,int))[:2]).ravel()
    scale=np.sqrt(np.diag(expected_covariance))
    np.testing.assert_allclose(fields.mean(0),expected_mean,atol=4*scale.max()/np.sqrt(len(fields)))
    np.testing.assert_allclose(np.cov(fields.T),expected_covariance,atol=.15*scale.max()**2)
    # The field is pinned near the images and free far from them: the farthest point keeps the prior variance.
    assert expected_covariance[0,0]<.5*toy['sigma']**2 and abs(np.var(fields[:,4])/toy['sigma']**2-1)<.2


def test_realizations_with_low_order_modes_removed_match_the_projected_covariance_at_the_images():
    modes=('translation','convergence','shear')
    toy=_toy_conditioning(modes)
    fields=_realizations(toy,toy['xy'],modes,1200)
    cross=toy['k']*toy['w2'][None]
    expected_covariance=toy['k']-cross@np.linalg.solve(toy['c'],cross.T)
    expected_mean=toy['k']@(toy['w2']*toy['v'].reshape(-1))
    scale=np.sqrt(np.diag(toy['k']).max())
    np.testing.assert_allclose(fields.mean(0),expected_mean,atol=4*scale/np.sqrt(len(fields))+1e-3)
    np.testing.assert_allclose(np.cov(fields.T),expected_covariance,atol=.15*scale**2)
    q=np.asarray(gp_knee.low_order_projector(toy['layout']['xy'],toy['layout']['mask'],modes)[0])[0]
    assert np.abs(fields@q).max()<2e-3*scale*np.sqrt(10)          # no translation, convergence or shear over the images


def test_a_realization_is_fixed_by_its_generator_and_its_jacobian_is_the_gradient_of_its_deflection():
    from kappaccino.plotting.grf import FieldRealization,draw_field
    modes=('translation','convergence','shear')
    toy=_toy_conditioning(modes)
    build=lambda:FieldRealization(toy['layout'],toy['tables'],draw_field(toy['layout'],toy['tables'],toy['cholesky'],toy['v'],toy['sigma'],
        toy['lam'],4.,modes=modes,rng=np.random.default_rng([3,17]),n_modes=256),np.array([1.,.5]),modes=modes)
    layout=dict(toy['layout'],block_of_family=jnp.array([0,0]));toy['layout']=layout
    field,again=build(),build()
    points=np.array([[.3,-.4],[1.7,2.2],[-4.,3.]]);families=np.array([0,1,1])
    values=np.asarray(field(*points.T,families))
    np.testing.assert_array_equal(values,np.asarray(again(*points.T,families)))
    step=1e-5
    for axis in range(2):
        shift=np.eye(2)[axis]*step
        numeric=(np.asarray(field(*(points+shift).T,families))[:2]-np.asarray(field(*(points-shift).T,families))[:2])/(2*step)
        np.testing.assert_allclose(numeric,values[[2+axis,4+axis]],atol=2e-6)
    np.testing.assert_allclose(values[3],values[4],atol=1e-10)   # curl-free
    np.testing.assert_allclose(np.asarray(field(points[1:2,0],points[1:2,1],[0]))/values[:,1:2],2.,rtol=1e-12)   # family weight


def test_jax_evaluation_matches_the_scipy_hermite_pieces(monkeypatch):
    """The compiled evaluation uses power-basis pieces of the same quintic Hermite interpolant."""
    smooth,_=toy_field(monkeypatch)
    rng=np.random.default_rng(5)
    points=rng.uniform(-4,4,(5000,2)); points[0]=smooth.xy[0,1]     # one point exactly on an image
    families=rng.integers(0,2,len(points))
    dx,dy,j00,j01,j10,j11=smooth(points[:,0],points[:,1],families)
    xy=smooth.xy[0];u=smooth.u[0]
    d=points[:,None]-xy[None];r=np.linalg.norm(d,axis=-1);safe=np.maximum(r,1e-10)
    f=np.where(r>1e-10,smooth.ct1(r)/safe,smooth.ct2(0.));g=np.where(r>1e-10,(smooth.ct2(r)-smooth.ct1(r)/safe)/safe**2,0.)
    du=np.einsum('pmi,mi->pm',d,u)
    value=np.einsum('pm,ma->pa',smooth.ct(r),u)+np.einsum('pm,pma->pa',f*du,d)
    deriv=(np.einsum('pm,ma,pmc->pac',f,u,d)+np.einsum('pm,pma,mc->pac',f,d,u)
           +np.einsum('pm->p',f*du)[:,None,None]*np.eye(2)+np.einsum('pm,pma,pmc->pac',g*du,d,d))
    weight=smooth.weights[families]
    np.testing.assert_allclose(np.column_stack([dx,dy]),smooth.amplitude*value*weight[:,None],atol=1e-12)
    np.testing.assert_allclose(np.stack([j00,j01,j10,j11],-1).reshape(-1,2,2),smooth.amplitude*deriv*weight[:,None,None],rtol=0,atol=1e-9)
