=================================================
Abell 370 CANUCS lensing reconstruction
=================================================

Author: Rachel Gledhill

BRIEF DESCRIPTION OF THE METHOD

The lens model was constrained with parametric lens modelling code Lenstool (Kneib et al. 1996; Jullo et al. 2007; Jullo & Kneib 2009) and is described in Gledhill et al. (2025), https://doi.org/10.3847/1538-4357/ad684a . The cluster member catalog was derived from MUSE observations from Lagattuta et al. (2019). The model was constrained with  24 secure multiple image systems with spectroscopic redshifts. The catalog is based on existing catalogs from Strait et al. (2018) and Lagattuta et al. (2019) and includes new spectroscopic redshifts for three systems and new multiply-imaged clumps.

COSMOLOGY

Omega_matter=0.3
Omega_Lambda=0.7
H_0=70 km/s/Mpc


DELIVERABLES
We provide the following lensing maps convergence maps (kappa), shear maps (gamma) with both components (gamma1, gamma2) and deflection maps (x and y components) given in arcsec (dpl1, dpl2). All maps are given for Dds/Ds=1 (coordinate of the centre 39.969773, -1.572130). We provide best-fit lens model maps in 

-50 mas resolution, 200 arcsec size (dlsds1-best-50mas)
-100 mas resolution, 500 arcsec size (dlsds1-best-100mas)
-300 mas resolution, 1200 arcsec size (dlsds1-best-300mas).

We also provide lensing maps for 100 Bayesian lens model samples. They are provided as multi-extension fits (MEF) files with one primary hdu and 100 extensions, corresponding to 100 randomly selected lens models from the MCMC chain. They are provided in

-200 mas resolution, 500 arcsec size (dlsds1-samples-200mas)
-600 mas resolution, 1200 arcsec size (dlsds1-samples-600mas)

We also provide Lenstool input parameter file used for optimisation (lenstool-param), Lenstool best-fit parameter file (lenstool-bestparam), multiple image catalog (lenstool-multim) and cluster member catalog (lenstool-clugal) in Lenstool format.

