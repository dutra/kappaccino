UNCOVER Abell 2744 lens model
-----------------------------

Authors: Lukas J. Furtak and Adi Zitrin
Version: 2.0
Created: July 26, 2024
Model reference papers: Furtak et al. (2023) & Price et al. (in prep.)
Contact: furtak@post.bgu.ac.il

Updates compared to previous version
------------------------------------
We added 13 spectroscopic redshifts from JWST/NIRSpec MSA and JWST/NIRCam grism observations, and 132 additional cluster members selected from JWST/NIRCam imaging. The additional spectroscopic redshifts in particular constrain the northern and north-western sub-structures that were previously mostly constrained with photometric redshifts.


Modeling method and constraints
-------------------------------
The strong lensing (SL) model is constructed with the UNCOVER JWST data (Bezanson et al. 2024) using a the updated analytic lens modeling code by Zitrin et al. (2015) that was recently used in various studies with HST and JWST data (e.g. Pascale et al. 2022; Meena et al. 2023).

The SL model contains the compilation of spectroscopic cluster members and SL features from Bergamini et al. (2023), and references therein, in the main cluster core, and adds newly discovered multiple images from the UNCOVER imaging and spectroscopy in the northern and north-western extended DM structures previously revealed by weak lensing (WL) studies. We also added more cluster member galaxies from the BUFFALO HST imaging (Steinhardt et al. 2020) and Mega Science JWST imaging (Suess et al. 2024).
We model the cluster with five PIEMD cluster-scale dark-matter (DM) halos entered on the five BCGs in the main cluster core and the northern and north-western sub-structures, and 552 cluster member galaxies. The model is constrained by a total of 141 multiple images belonging to 48 sources.

The model has a lens plane RMS of 0.60 arcsec.


Release contents
----------------
In this release, we supply WCS-registered FITS maps of the deflection field (in pixels & arc-seconds), kappa, gamma, and the lensing potential normalized to Dds/Ds=1. We also supply magnification maps for various source redshifts: 1, 2, 4, 6, 8, 10, 15 and 20. Note that the lensing signal is not particularly sensitive to source redshift beyond z~6. The lensing maps have a FoV of 7.6' x 7.6', which covers the entire central UNCOVER field. We provide maps for the best-fit model in a resolution of 0.1''/pix and 0.04''/pix. The release also comprises a range of 50 maps drawn from our final MCMC chain for uncertainty computation in the 0.1''/pix resolution. Please contact us if you need any other lensing product (individual magnifications, time delays, source planes, custom maps etc.).


Directory structure
-------------------
Best-model_low-resolution_40mas/                --> Best-fit model maps in 0.04''/pix resolution
Best-model_low-resolution_100mas/               --> Best-fit model maps in 0.1''/pix resolution
Model-range_low-resolution_100mas/              --> Error range maps in 0.1''/pix resolution
UNCOVER_SL-map_FOV.reg                          --> DS9 region-file of the lens-map FoV for visualisation purposes
A2744_multiple-images_v2.0.reg                  --> DS9 region file of all multiple images used to constrain the model
A2744_cluster-members_v2.0.fits                 —-> .fits table containing the full catalog of HST and JWST selected cluster member galaxies used in this model
A2744_multiple-images_v2.0.fits                 --> .fits table containing the full catalog of multiple images used to constrain the model
Full-cluster_lensing_grid_critcurves_v2.0.pdf   --> Color image of the cluster with overlaid critical lines and multiple image positions


References
----------
Bergamini et al. (2023), A&A, 670, 15
Bezanson et al. (2042), arXiv:2212.04026
Furtak et al. (2023), MNRAS, 523, 4568
Meena et al. (2023), ApJL, 944, 9
Pascale et al. (2022), ApJ, 938, 10
Steinhardt et al. (2020), ApJS, 247, 20
Suess et al. (2024), arXiv:2404.13132
Zitrin et al. (2015), ApJ, 801, 21