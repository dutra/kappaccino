=================================================
MACS J1149.5+2223 CANUCS lensing reconstruction
=================================================

Author: Gregor Rihtarsic

BRIEF DESCRIPTION OF THE METHOD

The lens model was constrained with parametric lens modelling code Lenstool (Kneib et al. 1996; Jullo et al. 2007; Jullo & Kneib 2009). This is a preliminary version of the lens model available at the time of the CANUCS DR1. The final lens model (version v2) will be released after the publication (Rihtarsic et al, in prep.). The model parametrization is based on Jauzac et al. (2016) and Desprez et al. (2018) lens models with additional X-ray mass component. The cluster member catalog was taken from Schuldt et al. (2024). The model was constrained with the gold catalog of 30 multiple image systems with spectroscopic redshifts, derived from existing MUSE catalogs (Schuldt et al., 2024) and CANUCS NIRCam imaging and NIRSpec and NIRISS spectroscopy. 

COSMOLOGY

Omega_matter=0.3
Omega_Lambda=0.7
H_0=70 km/s/Mpc


DELIVERABLES
We provide the following lensing maps convergence maps (kappa), shear maps (gamma) with both components (gamma1, gamma2) and deflection maps (x and y components) given in arcsec (dpl1, dpl2). All maps are given for Dds/Ds=1 (coordinate of the centre 177.398734, 22.398520). We provide best-fit lens model maps in 

-50 mas resolution, 200 arcsec size (dlsds1-best-50mas)
-100 mas resolution, 500 arcsec size (dlsds1-best-100mas)
-300 mas resolution, 1200 arcsec size (dlsds1-best-300mas).

We also provide lensing maps for 100 Bayesian lens model samples. They are provided as multi-extension fits (MEF) files with one primary hdu and 100 extensions, corresponding to 100 randomly selected lens models from the MCMC chain. They are provided in

-200 mas resolution, 500 arcsec size (dlsds1-samples-200mas)
-600 mas resolution, 1200 arcsec size (dlsds1-samples-600mas)

We also provide the cluster member catalog in Lenstool format (lenstool-clugal) - the magnification of cluster members was set to 1 in the photometric catalogs. Lenstool parameter files and input multiple image catalogs will be available with version v2 upon the lens model publication.
