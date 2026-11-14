"""Normalization of the illustrative, scatter-free input mass distribution."""
import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc

from kappaccino.mock_validation.aggregate_plots import _schechter_curve


def test_reference_normalization_and_generator_cdf():
    selection = dict(schechter_alpha=-0.7, dpie_mass_anchor_msun=8e11, mass_luminosity_exponent=1.2,
                     schechter_magnitude=19.82, pivot_magnitude=19.82)
    lower, upper = 0.02, 20.0
    mass = lambda ell: np.log10(8e11) + 1.2 * np.log10(ell)
    density = lambda x: float(_schechter_curve(selection, np.asarray(x), 200, (0.02, upper)))
    np.testing.assert_allclose(quad(density, mass(lower), mass(upper), epsabs=1e-9)[0], 200, rtol=1e-12)
    # Independent incomplete-gamma CDF used by the generator.
    shape = 0.3
    expected = 200 * (gammainc(shape, 1) - gammainc(shape, lower)) / (gammainc(shape, upper) - gammainc(shape, lower))
    np.testing.assert_allclose(quad(density, mass(lower), mass(1))[0], expected, rtol=1e-12)
    np.testing.assert_array_equal(_schechter_curve(selection, np.array([mass(lower) - 1, mass(upper) + 1]), 200, (0.02, upper)), 0)
    # Evaluating a smaller plotting range must not renormalize the curve.
    np.testing.assert_allclose(density(mass(1)), _schechter_curve(selection, np.array([mass(1), mass(2)]), 200, (0.02, upper))[0])


def test_missing_selection_does_not_invent_reference():
    assert _schechter_curve({}, np.array([10.0]), 200) is None


def test_cutoff_magnitude_shifts_draw_and_curve_together():
    """Members drawn with schechter_magnitude != pivot_magnitude follow the plotted curve."""
    from kappaccino.mock_validation.truth import truncated_schechter_luminosities
    alpha, lower, upper = -0.8, 0.02, 20.0
    cutoff = 10 ** (-0.4 * (19.0 - 19.82))
    rng = np.random.default_rng(1)
    draws = cutoff * truncated_schechter_luminosities(rng, 200_000, alpha=alpha, lower=lower / cutoff, upper=upper / cutoff)
    assert lower <= draws.min() and draws.max() <= upper
    selection = dict(schechter_alpha=alpha, dpie_mass_anchor_msun=1.0, mass_luminosity_exponent=1.0,
                     schechter_magnitude=19.0, pivot_magnitude=19.82)
    density = lambda x: float(_schechter_curve(selection, np.asarray(x), 1.0, (lower, upper)))
    np.testing.assert_allclose(quad(density, np.log10(lower), np.log10(upper), epsabs=1e-10)[0], 1.0, rtol=1e-8)
    # Fraction brighter than the pivot L*: drawn against the curve.
    expected = quad(density, 0.0, np.log10(upper), epsabs=1e-10)[0]
    np.testing.assert_allclose((draws > 1.0).mean(), expected, atol=4e-3)
