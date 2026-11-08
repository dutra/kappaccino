"""Deprecated alias of :mod:`kappaccino` (the package's former name).

``import lenscluster`` and ``import lenscluster.<module>`` return the corresponding ``kappaccino`` modules, the same
module objects, so code written against the old name and configurations pickled by older runs keep working. A
``FutureWarning`` is raised on import; use ``kappaccino`` in new code. The alias will be removed in a future release.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings

_OLD, _NEW = "lenscluster", "kappaccino"

warnings.warn("lenscluster has been renamed kappaccino; import kappaccino instead", FutureWarning, stacklevel=2)


class _AliasLoader(importlib.abc.Loader):
    """Loads ``lenscluster.<name>`` as the already imported ``kappaccino.<name>``."""

    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.original_spec = module.__spec__
        return module

    def exec_module(self, module) -> None:
        # The import system has just set the alias spec on the shared module object; restore the module's own spec so
        # that kappaccino's relative imports and __package__ are unchanged.
        module.__spec__ = self.original_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves every ``lenscluster.*`` module name to the ``kappaccino.*`` module of the same path."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        new = _NEW + fullname[len(_OLD):]
        if importlib.util.find_spec(new) is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(new), is_package=True)


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

import importlib.util  # noqa: E402  (find_spec above)

_package = importlib.import_module(_NEW)
__path__ = []                        # a package, so that submodules resolve through the finder above
__all__ = getattr(_package, "__all__", [])
globals().update({name: getattr(_package, name) for name in dir(_package) if not name.startswith("__")})
