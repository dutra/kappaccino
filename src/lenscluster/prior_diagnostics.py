"""Normalized scalar priors for posterior diagnostics in sampling coordinates."""
from . import priors


def saved_prior(spec):
    """The normalized sampling-coordinate prior of ``spec`` as a frozen scipy.stats distribution (distribution_for_site)."""
    return priors.to_scipy(spec.latent_prior)
