from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kappaccino.io.image_catalog import CatalogError, load_image_catalog, write_image_catalog
from kappaccino.partition.slots import association_log_weights, claim_columns, claimed_groups


def test_image_and_knot_priors(tmp_path):
    path = write_image_catalog(tmp_path / "images.yaml", {
        "1": {"z": 2.0, "z_err": 0.01, "images": {"a": {"ra": 10.0, "dec": 1.0}, "b": {"ra": 10.001, "dec": 1.0, "prior": 0.3}}},
        "2": {"z": 0.0, "knots": ["A", "B"], "images": {
            "a": {"prior": 0.5, "knots": {"A": [10.002, 1.0], "B": {"ra": 10.003, "dec": 1.0, "prior": 0.2}}},
            "b": {"knots": {"A": [10.004, 1.0], "B": [10.005, 1.0]}}}},
    }, defaults={"prior": 0.9})
    table = load_image_catalog(path).set_index("image_label")
    assert table.loc["1.a", "association_prior"] == 0.9                     # defaults
    assert table.loc["1.b", "association_prior"] == 0.3
    assert table.loc["2A.a", "association_prior"] == 0.5                    # the image's prior for its knots
    assert table.loc["2B.a", "association_prior"] == 0.2                    # a knot's own
    assert table.loc["1.a", "catalog_z_err"] == 0.01 and np.isnan(table.loc["2A.a", "catalog_z_err"])


def test_prior_out_of_range(tmp_path):
    path = tmp_path / "images.yaml"
    path.write_text("format: lenscluster-images/1\nsystems:\n  '1':\n    z: 2.0\n    images:\n"
                    "      a: {ra: 10.0, dec: 1.0, prior: 1.5}\n      b: {ra: 10.1, dec: 1.0}\n")
    with pytest.raises(CatalogError, match="prior"):
        load_image_catalog(path)


def test_association_weights():
    weights = np.array([0.5, 0.25, 0.25])            # unrelated, one trusted family, one new slot
    table = pd.DataFrame({"claimed_family": ["F1", "X", "X", ""], "association_prior": [0.8, 0.1, 0.1, np.nan]})
    claimed = claimed_groups(table, ["F1"])
    assert claimed == [(1, 2)]
    columns = claim_columns(table, ["F1"], claimed)
    assert columns.tolist() == [1, 2, 2, -1]
    out = np.exp(association_log_weights(weights, columns, table["association_prior"].to_numpy()))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[0], [0.2 * 0.5 / 0.75, 0.8, 0.2 * 0.25 / 0.75])
    assert out[1, 2] == pytest.approx(0.1)
    np.testing.assert_allclose(out[3], weights)       # no claim: the shared weights
