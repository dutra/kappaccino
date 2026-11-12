"""The lens profiles the solver knows, by Lenstool-style code.

Each entry ties a code to its kernel module (``kappaccino.lens.profiles.<name>``) and to what the model builder needs:
the fields a potential definition must carry and which polar pair is sampled as a Cartesian pair.  Every kernel module
has the same interface:

``PARAMETERS``
    Names of the ``PackedLensState`` fields the kernel takes, in order.
``fields(x, y, *parameters, derivative=True)``
    One component at points ``x``, ``y`` (N,): the deflection ``alpha`` (N, 2), the Jacobian ``U = d alpha / d theta``
    (N, 2, 2) and its derivative ``dU[..., i, j, l] = d U_ij / d theta_l`` (N, 2, 2, 2), or None for ``dU`` when
    ``derivative`` is False.  A normalization may be a scalar or one value per point.
``grouped_fields[derivative]``
    The same over a batch of components: every parameter gains a leading component axis.

Adding a profile means a kernel module with this interface, an entry here, and its packing in
``Evaluator.packed_state``.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from ...core.model import CORED_POWER_LAW_PROFILE, DP_IE_PROFILE, MASS_SHEET_PROFILE, NFW_PROFILE, SHEAR_PROFILE
from . import cored_power_law, dpie, nfw, shear, sheet

_HALO_FIELDS = ("x_centre", "y_centre", "ellipticite", "angle_pos", "core_radius_kpc", "cut_radius_kpc", "v_disp")


@dataclass(frozen=True)
class LensProfile:
    code: int
    label: str
    kernel: ModuleType                          # PARAMETERS, fields and grouped_fields (see the module docstring)
    required_fields: tuple[str, ...]            # numeric fields of a potential definition, besides id, profil and z_lens
    polar_shape: tuple[str, str] | None         # (amplitude, angle) pair sampled together ...
    cartesian_shape: tuple[str, str] | None     # ... as this Cartesian pair

    @property
    def parameters(self) -> tuple[str, ...]:
        """Names of the ``PackedLensState`` fields the kernel takes, in order."""
        return self.kernel.PARAMETERS

    def grouped_fields(self, x, y, packed, components, *, derivative: bool):
        """(alpha, U, dU or None) of the components ``components`` (static indices into ``packed``), shapes (C, N, ...)."""
        import jax.numpy as jnp
        index = jnp.asarray(components)
        arguments = [jnp.take(jnp.asarray(getattr(packed, name), dtype=jnp.float64), index, axis=0) for name in self.parameters]
        return self.kernel.grouped_fields[bool(derivative)](x, y, *arguments)


PROFILES: dict[int, LensProfile] = {
    DP_IE_PROFILE: LensProfile(
        code=DP_IE_PROFILE, label="dPIE", kernel=dpie, required_fields=_HALO_FIELDS,
        polar_shape=("ellipticite", "angle_pos"), cartesian_shape=("e1", "e2")),
    NFW_PROFILE: LensProfile(
        code=NFW_PROFILE, label="elliptical NFW", kernel=nfw,
        required_fields=("x_centre", "y_centre", "ellipticite", "angle_pos", "log10_m200", "scale_radius_kpc"),
        polar_shape=("ellipticite", "angle_pos"), cartesian_shape=("e1", "e2")),
    SHEAR_PROFILE: LensProfile(
        code=SHEAR_PROFILE, label="external shear", kernel=shear, required_fields=("gamma", "angle_pos"),
        polar_shape=("gamma", "angle_pos"), cartesian_shape=("gamma1", "gamma2")),
    MASS_SHEET_PROFILE: LensProfile(
        code=MASS_SHEET_PROFILE, label="external convergence", kernel=sheet,
        required_fields=("kappa",), polar_shape=None, cartesian_shape=None),
    # Members only (MemberPopulationConfig.profile_type): built by the scaling relation, never from a potential definition.
    CORED_POWER_LAW_PROFILE: LensProfile(
        code=CORED_POWER_LAW_PROFILE, label="cored power law", kernel=cored_power_law, required_fields=_HALO_FIELDS,
        polar_shape=("ellipticite", "angle_pos"), cartesian_shape=("e1", "e2")),
}
SUPPORTED_PROFILES = frozenset(PROFILES)
ELLIPTICAL_PROFILES = frozenset({DP_IE_PROFILE, NFW_PROFILE, CORED_POWER_LAW_PROFILE})


def profile(code: int) -> LensProfile:
    """The profile of a code; ``ValueError`` for an unknown one."""
    try:
        return PROFILES[int(code)]
    except KeyError:
        raise ValueError(f"Unsupported lens profile code {code}; known: {sorted(PROFILES)}") from None
