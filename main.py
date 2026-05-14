"""
main.py
-------
Entry point for the 2D Poisson FD solver pipeline.

Runs:
  1. Convergence study
  2. VTK export (2D + 3D modes)
  3. HDF5 export
  4. Static figures
  5. Animated GIFs (convergence + 3D rotation)

Usage
-----
    python src/main.py

All outputs are written to ./outputs/.

Author  : M1 CHPS – UPVD 2025-2026
Supervisor: Serge Dumont  <https://perso.univ-perp.fr/sdumont/>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── make sure src/ is on the path when called from the project root ──────────
sys.path.insert(0, str(Path(__file__).parent))

from poisson2d.solver import (
    build_grid,
    build_laplacian,
    build_rhs,
    node_coords,
    solve_direct,
    convergence_study,
    estimate_convergence_rate,
)
from poisson2d.vtk_io import write_vtk
from poisson2d.hdf5_io import write_hdf5
from poisson2d.plotting import (
    plot_solution,
    plot_solution_3d,
    plot_convergence,
    plot_cpu_time,
    plot_error_and_exact,
    make_convergence_gif,
    make_rotation_gif,
)

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
# Test case 1: f = sin(2πx)sin(2πy),  exact solution known
# ---------------------------------------------------------------------------

def f_sin(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def u_sin_exact(x, y):
    return (1.0 / (8 * np.pi**2)) * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


# Test case 2: uniform source f = 1  (no closed-form exact solution)
def f_const(x, y):
    return np.ones_like(x)


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

OUT = Path("outputs")
VTK_DIR   = OUT / "vtk"
HDF5_DIR  = OUT / "hdf5"
FIG_DIR   = OUT / "figures"
GIF_DIR   = OUT / "gif"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_test_case_1(N_hires: int = 99) -> None:
    """Full pipeline for the sin/sin test case."""
    print("\n" + "=" * 60)
    print("  Test case 1: f = sin(2πx)sin(2πy)")
    print("=" * 60)

    # ── High-resolution solution ───────────────────────────────────────────
    nodes, h = build_grid(N_hires)
    x, y = node_coords(nodes, h)
    A = build_laplacian(N_hires, h)
    B = build_rhs(f_sin, nodes, h)
    U = solve_direct(A, B)
    Uex = u_sin_exact(x, y)

    # ── VTK export ─────────────────────────────────────────────────────────
    write_vtk(VTK_DIR / "poisson_sin_2d.vtk", N_hires, {"U": U, "Uex": Uex}, "Poisson2D sin/sin", mode="2d")
    write_vtk(VTK_DIR / "poisson_sin_3d.vtk", N_hires, {"U": U, "Uex": Uex}, "Poisson2D sin/sin", mode="3d")

    # ── Static figures ─────────────────────────────────────────────────────
    plot_solution(N_hires, U, title="Numerical solution  U  (sin/sin)", save_path=FIG_DIR / "solution_2d.png")
    plot_solution_3d(N_hires, U, title="Numerical solution  U  – 3D view", save_path=FIG_DIR / "solution_3d.png")
    plot_error_and_exact(N_hires, U, Uex, save_path=FIG_DIR / "solution_comparison.png")

    # ── Convergence study ──────────────────────────────────────────────────
    N_vals = [4, 9, 19, 49, 99, 199]
    metrics = convergence_study(N_vals, f_sin, u_sin_exact)

    alpha, C = estimate_convergence_rate(metrics["h"], metrics["error_l2"])
    print(f"\n  Convergence rate  α ≈ {alpha:.4f}  (expected ≈ 2.00)")
    print(f"  Constant          C ≈ {C:.4e}\n")

    plot_convergence(metrics["h"], metrics["error_l2"], save_path=FIG_DIR / "convergence.png")
    plot_cpu_time(metrics["N"], metrics["cpu_time"], save_path=FIG_DIR / "cpu_time.png")

    # ── HDF5 export ────────────────────────────────────────────────────────
    write_hdf5(HDF5_DIR / "poisson_sin.hdf5", N_vals, f_sin, u_sin_exact)

    # ── Animated GIFs ──────────────────────────────────────────────────────
    gif_N_vals = [4, 9, 19, 39, 59, 79, 99]
    make_convergence_gif(gif_N_vals, f_sin, u_sin_exact, gif_path=GIF_DIR / "convergence.gif", fps=1)
    make_rotation_gif(N_hires, U, gif_path=GIF_DIR / "rotation_3d.gif", n_frames=36, fps=12)


def run_test_case_2(N_hires: int = 99) -> None:
    """Pipeline for the uniform-source test case (f=1)."""
    print("\n" + "=" * 60)
    print("  Test case 2: f = 1  (uniform source)")
    print("=" * 60)

    nodes, h = build_grid(N_hires)
    x, y = node_coords(nodes, h)
    A = build_laplacian(N_hires, h)
    B = build_rhs(f_const, nodes, h)
    U = solve_direct(A, B)

    write_vtk(VTK_DIR / "poisson_const_2d.vtk", N_hires, {"U": U}, "Poisson2D f=1", mode="2d")
    write_vtk(VTK_DIR / "poisson_const_3d.vtk", N_hires, {"U": U}, "Poisson2D f=1", mode="3d")

    plot_solution(N_hires, U, title="Numerical solution  U  (f=1)", save_path=FIG_DIR / "solution_const_2d.png")
    plot_solution_3d(N_hires, U, title="Numerical solution  U  – 3D view  (f=1)", save_path=FIG_DIR / "solution_const_3d.png")
    make_rotation_gif(N_hires, U, gif_path=GIF_DIR / "rotation_const_3d.gif", n_frames=36, fps=12)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_test_case_1(N_hires=99)
    run_test_case_2(N_hires=99)
    print("\n✔  All outputs written to ./outputs/")
