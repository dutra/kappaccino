"""Regenerate the scaling-relation figure from saved physical draws and member summaries."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kappaccino.plotting.plots import plot_scaling_relation
from kappaccino.io.hdf5 import load_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    specs = [SimpleNamespace(**p) for p in json.loads((run / "init/artifacts/state.json").read_text())["parameter_specs"]]
    data = load_arrays(run / "final/artifacts/posterior.h5")
    physical = data["physical"]
    table = pd.read_csv(run / "final/tables/member_scaling_relation.csv")
    # Check the saved table and posterior use the same normalizations and slopes.
    names = {spec.name: i for i, spec in enumerate(specs)}
    for population, members in table.groupby("potfile_id"):
        velocity = physical[:, names[f"{population}.sigma"], None] * members.luminosity_ratio.to_numpy()[None, :] ** physical[:, names[f"{population}.alpha_sigma"], None]
        np.testing.assert_allclose(np.median(velocity, axis=0), members.v_disp_scaling_median, rtol=1e-12)
    config = json.loads((run / "final/artifacts/config.json").read_text())
    prefix = config["paths"].get("figure_prefix")
    output = run / "final/figures" / ((f"{prefix}_" if prefix else "") + "scaling_relation.pdf")
    plot_scaling_relation(table, physical, specs, output)
    print(f"Verified saved velocity relation for {len(table)} members; regenerated {output}")


if __name__ == "__main__":
    main()
