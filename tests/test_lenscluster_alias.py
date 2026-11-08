"""The deprecated ``lenscluster`` name resolves to the ``kappaccino`` modules themselves."""
from __future__ import annotations

import pickle
import warnings

import pytest


def test_old_name_imports_the_same_modules_with_a_warning():
    import importlib
    import sys

    sys.modules.pop("lenscluster", None)
    with pytest.warns(FutureWarning, match="renamed kappaccino"):
        importlib.import_module("lenscluster")
    import kappaccino.config
    import kappaccino.mock_validation
    import lenscluster.config
    import lenscluster.mock_validation

    assert lenscluster.config is kappaccino.config
    assert kappaccino.config.__spec__.name == "kappaccino.config"       # the alias leaves the module's own spec alone
    assert lenscluster.mock_validation.run_mock_validation is kappaccino.mock_validation.run_mock_validation


def test_configs_pickled_under_the_old_name_load():
    from kappaccino.config import RunPathsConfig

    # Protocol 0 stores module names as text lines, so the old name can be substituted without breaking the framing.
    payload = pickle.dumps(RunPathsConfig(output_dir="results", run_name="demo"), protocol=0).replace(b"kappaccino.config", b"lenscluster.config")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)          # the alias warns on its first import only
        loaded = pickle.loads(payload)
    assert type(loaded) is RunPathsConfig and loaded.run_name == "demo"
