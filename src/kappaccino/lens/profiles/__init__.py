"""Lens profiles: one kernel module per profile with a common interface, and the registry that maps codes to them.

=====================  ====  =============================================================
module                 code  profile
=====================  ====  =============================================================
``dpie``                 81  dPIE, closed form in the Cartesian ellipticity
``cored_power_law``     102  dPIE with a free 3D slope (members), Gauss-Legendre quadrature
``nfw``                  12  elliptical NFW, CSE sum of Oguri (2021)
``shear``                14  external shear
``sheet``               101  uniform mass sheet
=====================  ====  =============================================================

``frame`` holds the principal-frame helpers the rotated kernels share, and ``registry`` the interface every kernel
module follows.
"""
from . import cored_power_law, dpie, frame, nfw, shear, sheet
from .registry import ELLIPTICAL_PROFILES, PROFILES, SUPPORTED_PROFILES, LensProfile, profile

__all__ = ["ELLIPTICAL_PROFILES", "PROFILES", "SUPPORTED_PROFILES", "LensProfile", "profile",
           "cored_power_law", "dpie", "frame", "nfw", "shear", "sheet"]
