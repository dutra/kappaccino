"""Arrays in HDF5 files: the format of every array a run saves.

One dataset per name at the file root.  Unicode arrays are stored as UTF-8 strings and read back as ``str`` arrays;
None values are not written, and :func:`load_arrays` simply has no entry for them.  Only plain numerical, boolean and
string arrays are stored: no Python objects, no pickles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

STRING = h5py.string_dtype("utf-8")


def save_arrays(path: str | Path, **arrays: Any) -> Path:
    """Write arrays to an HDF5 file, one dataset per keyword.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file (``.h5``); parent directories are created and an existing file is replaced.
    **arrays
        Array-likes and scalars by dataset name.  None values are skipped.  Arrays of more than one element are
        gzip-compressed.

    Returns
    -------
    pathlib.Path
        ``path``.

    Raises
    ------
    TypeError
        If a value is an object array (a pickle would be needed to store it).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        for name, value in arrays.items():
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.kind == "U":
                file.create_dataset(name, data=array.astype(object), dtype=STRING)
                continue
            if array.dtype.kind == "O":
                raise TypeError(f"{name}: object arrays cannot be saved; convert to numbers or strings")
            options = {"compression": "gzip", "shuffle": True} if array.ndim and array.size > 1 else {}
            file.create_dataset(name, data=array, **options)
    return path


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Read every dataset of a file written by :func:`save_arrays`.

    Parameters
    ----------
    path : str or pathlib.Path
        The HDF5 file.

    Returns
    -------
    dict of str to numpy.ndarray
        Arrays by dataset name (0-d arrays for scalars); string datasets as ``str`` arrays.
    """
    with h5py.File(Path(path), "r") as file:
        out = {}
        for name, dataset in file.items():
            if h5py.check_string_dtype(dataset.dtype) is not None:
                out[name] = np.asarray(dataset.asstr()[()], dtype=str)
            else:
                out[name] = np.asarray(dataset[()])
        return out
