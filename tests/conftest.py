from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import numpyro.distributions as dist
import pytest

jax.config.update("jax_enable_x64", True)

from kappaccino.priors import lognormal_from_moments
from kappaccino.config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                                MemberPopulationConfig, RunPathsConfig, SamplingConfig, SearchConfig)




def write_offset_catalog(path: Path, rows, reference=(0.0, 0.0)) -> Path:
    """YAML image catalog from rows of (label "F.i", x, y, z): offsets in arcsec from ``reference`` (ra0, dec0), every
    image at magnitude 24 +- 0.05."""
    from kappaccino.io.image_catalog import systems_from_offsets, write_image_catalog
    entries = [(*str(label).split("."), float(x), float(y), float(z), {"mag": 24.0, "mag_err": 0.05}) for label, x, y, z in rows]
    return write_image_catalog(path, systems_from_offsets(entries, reference))


def synthetic_cluster_config(root: Path) -> LensClusterSolverConfig:
    """A four-image, four-member toy cluster: two members near the images, two far away."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images = write_offset_catalog(root / "images.yaml", [("1.a", 3, 5, 2), ("1.b", -4, -5, 2), ("2.a", 8, -6, 3), ("2.b", -7, 9, 3)])
    members = root / "members.cat"
    members.write_text("#REFERENCE 3\ng1 2 3 1 .9 20 20 1 1\ng2 -6 7 1 .8 30 21 1 1\ng3 100 70 1 1 0 21 1 1\ng4 -100 -80 1 1 0 23 1 1\n")
    return LensClusterSolverConfig(
        model=LensModelConfig(
            cosmology=CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7),
            large_halos=(DPIEHaloConfig(id="halo", x_centre=0.0, y_centre=0.0, angle_pos=0.0, v_disp=600.0, core_radius_kpc=5.0,
                                        cut_radius_kpc=1000.0, z_lens=0.3, ellipticite=0.2,
                                        priors={"v_disp": dist.LogUniform(500.0, 700.0)}),),
            member_populations=(MemberPopulationConfig(id="members", catalog_path=str(members), mag0=20.0, corekpc=0.2, sigma=100.0,
                                                       cutkpc=10.0, z_lens=0.3,
                                                       sigma_prior=lognormal_from_moments(100.0, 20.0, low=50.0, high=200.0),
                                                       cutkpc_prior=lognormal_from_moments(10.0, 3.0, low=1.0, high=50.0)),),
            image_constraints=ImageConstraintsConfig(catalog_path=str(images))),
        search=SearchConfig(), sampling=SamplingConfig(),
        paths=RunPathsConfig(output_dir=str(root / "output"), run_name="toy"))


@pytest.fixture
def synthetic_config(tmp_path: Path) -> LensClusterSolverConfig:
    return synthetic_cluster_config(tmp_path)


def _brute_force_images(solver, packed, source, center, half, pitch, min_separation=0.01):
    """Every image of ``source`` in the square window ``center`` +- ``half``: the local minima of |beta - source| on a
    grid of the given pitch, polished by the solver's LM step and de-duplicated.  A reference search independent of the
    global finder's seeding."""
    axis_x = np.arange(center[0] - half, center[0] + half + 0.5 * pitch, pitch)
    axis_y = np.arange(center[1] - half, center[1] + half + 0.5 * pitch, pitch)
    gx, gy = np.meshgrid(axis_x, axis_y)
    bx, by, *_ = solver.ray_shoot_and_jacobian(packed, gx.ravel(), gy.ravel())
    distance = np.hypot(bx - source[0], by - source[1]).reshape(gx.shape)
    padded = np.pad(distance, 1, constant_values=np.inf)
    neighbours = [padded[1 + i:padded.shape[0] - 1 + i, 1 + j:padded.shape[1] - 1 + j]
                  for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
    minimum = np.all([distance <= n for n in neighbours], axis=0)
    polished = solver._refine(packed, float(source[0]), float(source[1]), gx[minimum], gy[minimum])
    x, y = polished["x"][polished["identification_pass"]], polished["y"][polished["identification_pass"]]
    keep = [i for i in range(x.size) if all(np.hypot(x[i] - x[j], y[i] - y[j]) > min_separation for j in range(i))]
    return x[keep], y[keep]


@pytest.fixture
def brute_force_images():
    return _brute_force_images
