"""
hdf5_io.py
----------
Save / load Poisson simulation results in HDF5 format (ViTables-compatible).

File layout
-----------
/                       root
├── N=10/
│   ├── U               (float64 array, N²)
│   ├── Uex             (float64 array, N²)
│   └── attrs: N, h, error_l2, cpu_time
├── N=20/ …
├── convergence/
│   ├── N               (int array)
│   ├── h               (float array)
│   ├── error_l2        (float array)
│   └── cpu_time        (float array)
└── attrs: alpha, C  (convergence-rate fit coefficients)

Author  : M1 CHPS – UPVD 2025-2026
Supervisor: Serge Dumont  <https://perso.univ-perp.fr/sdumont/>
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from .solver import (
    build_grid,
    build_laplacian,
    build_rhs,
    node_coords,
    solve_direct,
    l2_norm,
    estimate_convergence_rate,
)
import time


def write_hdf5(
    path: str | Path,
    N_values: list[int],
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    u_exact: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    overwrite: bool = True,
) -> dict[str, np.ndarray]:
    """Run simulations for all *N_values* and save to HDF5.

    Returns the convergence metrics dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "a"
    metrics: dict[str, list] = {"N": [], "h": [], "error_l2": [], "cpu_time": []}

    with h5py.File(path, mode) as fh:
        for N in N_values:
            nodes, h = build_grid(N)
            x, y = node_coords(nodes, h)

            t0 = time.perf_counter()
            A = build_laplacian(N, h)
            B = build_rhs(f, nodes, h)
            U = solve_direct(A, B)
            cpu = time.perf_counter() - t0

            grp = fh.require_group(f"N={N}")
            if "U" in grp:
                del grp["U"]
            grp.create_dataset("U", data=U, compression="gzip")

            err = float("nan")
            if u_exact is not None:
                Uex = u_exact(x, y)
                if "Uex" in grp:
                    del grp["Uex"]
                grp.create_dataset("Uex", data=Uex, compression="gzip")
                err = l2_norm(U - Uex, h)

            grp.attrs["N"] = N
            grp.attrs["h"] = h
            grp.attrs["error_l2"] = err
            grp.attrs["cpu_time"] = cpu

            metrics["N"].append(N)
            metrics["h"].append(h)
            metrics["error_l2"].append(err)
            metrics["cpu_time"].append(cpu)

            print(f"[HDF5] N={N:4d}  h={h:.4f}  ‖err‖_L2={err:.4e}  t={cpu:.3f}s")

        # Save convergence table
        conv = fh.require_group("convergence")
        for key, vals in metrics.items():
            arr = np.array(vals)
            if key in conv:
                del conv[key]
            conv.create_dataset(key, data=arr)

        # Convergence rate fit
        h_arr = np.array(metrics["h"])
        err_arr = np.array(metrics["error_l2"])
        valid = np.isfinite(err_arr) & (err_arr > 0)
        if valid.sum() >= 2:
            alpha, C = estimate_convergence_rate(h_arr[valid], err_arr[valid])
            fh.attrs["alpha"] = alpha
            fh.attrs["C"] = C
            print(f"[HDF5] Convergence rate  α ≈ {alpha:.4f}  C ≈ {C:.4e}")

    print(f"[HDF5] saved  {path}")
    return {k: np.array(v) for k, v in metrics.items()}


def read_hdf5(path: str | Path) -> dict:
    """Read all groups and datasets from an HDF5 file into a dict."""
    path = Path(path)
    out: dict = {}
    with h5py.File(path, "r") as fh:
        for key in fh.keys():
            item = fh[key]
            if isinstance(item, h5py.Group):
                out[key] = {
                    "attrs": dict(item.attrs),
                    "datasets": {k: item[k][()] for k in item.keys()},
                }
            else:
                out[key] = item[()]
        out["_file_attrs"] = dict(fh.attrs)
    return out
