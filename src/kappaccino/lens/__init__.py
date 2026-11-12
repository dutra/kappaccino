"""Lens physics in JAX: the lens profiles (``profiles``), cosmological distances (``cosmology``) and the far-field
series of distant members (``farfield``).  Everything here is double precision."""
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)
