"""
plotting.py
-----------
Matplotlib-based plots and animated GIFs for the Poisson solver results.

Author  : M1 CHPS – UPVD 2025-2026
Supervisor: Serge Dumont  <https://perso.univ-perp.fr/sdumont/>
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from solver import (
    build_grid,
    build_laplacian,
    build_rhs,
    node_coords,
    solve_direct,
    estimate_convergence_rate,
)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

CMAP = "RdYlBu_r"
FIGSIZE = (7, 5.5)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[FIG] saved  {path}")


# ---------------------------------------------------------------------------
# Solution field
# ---------------------------------------------------------------------------

def plot_solution(
    N: int,
    U: np.ndarray,
    title: str = "Numerical solution  U",
    save_path: str | Path | None = None,
) -> None:
    """Filled-contour plot of U on the interior grid."""
    nodes, h = build_grid(N)
    x, y = node_coords(nodes, h)

    xi = np.unique(x)
    yi = np.unique(y)
    Z = U.reshape(N, N)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cf = ax.contourf(xi, yi, Z, levels=40, cmap=CMAP)
    ax.contour(xi, yi, Z, levels=10, colors="k", linewidths=0.4, alpha=0.5)
    fig.colorbar(cf, ax=ax, label="u(x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")

    if save_path:
        _save(fig, Path(save_path))
    else:
        plt.show()


def plot_solution_3d(
    N: int,
    U: np.ndarray,
    title: str = "Numerical solution  U – 3D view",
    save_path: str | Path | None = None,
) -> None:
    """Surface plot of U."""
    nodes, h = build_grid(N)
    x, y = node_coords(nodes, h)

    xi = np.unique(x)
    yi = np.unique(y)
    Z = U.reshape(N, N)
    XX, YY = np.meshgrid(xi, yi)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XX, YY, Z, cmap=CMAP, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="u(x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(title)

    if save_path:
        _save(fig, Path(save_path))
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------

def plot_convergence(
    h_arr: np.ndarray,
    err_arr: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Log-log plot of L² error vs h with fitted convergence slope."""
    alpha, C = estimate_convergence_rate(h_arr, err_arr)
    h_fit = np.linspace(h_arr.min(), h_arr.max(), 200)
    err_fit = C * h_fit**alpha

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(h_arr, err_arr, "o-", color="#1f77b4", label="‖U − Uex‖_L²", ms=6)
    ax.loglog(
        h_fit,
        err_fit,
        "--",
        color="#d62728",
        label=rf"Fit: $C h^{{\alpha}}$,  $\alpha \approx {alpha:.2f}$",
    )
    ax.set_xlabel("h  (mesh size)")
    ax.set_ylabel("L² error")
    ax.set_title("Convergence of the FD scheme")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)

    if save_path:
        _save(fig, Path(save_path))
    else:
        plt.show()


def plot_cpu_time(
    N_arr: np.ndarray,
    time_arr: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Log-log CPU time vs N."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(N_arr, time_arr, "s-", color="#2ca02c", ms=6, label="CPU time")
    ax.set_xlabel("N  (interior nodes per direction)")
    ax.set_ylabel("CPU time  (s)")
    ax.set_title("Solver CPU time vs N")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()

    if save_path:
        _save(fig, Path(save_path))
    else:
        plt.show()


def plot_error_and_exact(
    N: int,
    U: np.ndarray,
    Uex: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Side-by-side: numerical vs exact solution and point-wise error."""
    nodes, h = build_grid(N)
    x, y = node_coords(nodes, h)
    xi, yi = np.unique(x), np.unique(y)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, data, label in zip(
        axes,
        [U, Uex, np.abs(U - Uex)],
        ["Numerical U", "Exact Uex", "|U − Uex|"],
    ):
        cf = ax.contourf(xi, yi, data.reshape(N, N), levels=40, cmap=CMAP)
        fig.colorbar(cf, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.suptitle(f"N = {N},  h = {h:.4f}", fontsize=12)
    fig.tight_layout()

    if save_path:
        _save(fig, Path(save_path))
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Animated GIF – convergence animation
# ---------------------------------------------------------------------------

def make_convergence_gif(
    N_values: list[int],
    f,
    u_exact,
    gif_path: str | Path,
    fps: int = 2,
) -> None:
    """Generate an animated GIF showing how the solution improves with N."""
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    tmp_dir = gif_path.parent / "_frames"
    tmp_dir.mkdir(exist_ok=True)

    for N in N_values:
        nodes, h = build_grid(N)
        x, y = node_coords(nodes, h)
        A = build_laplacian(N, h)
        B = build_rhs(f, nodes, h)
        U = solve_direct(A, B)
        Uex = u_exact(x, y)
        err = np.abs(U - Uex)

        xi, yi = np.unique(x), np.unique(y)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, data, lbl in zip(
            axes,
            [U, Uex, err],
            [f"Numerical  (N={N})", "Exact solution", f"|error|  max={err.max():.2e}"],
        ):
            cf = ax.contourf(xi, yi, data.reshape(N, N), levels=30, cmap=CMAP)
            fig.colorbar(cf, ax=ax, format="%.2e")
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

        fig.suptitle(
            rf"2D Poisson – FD convergence  $h = {h:.4f}$", fontsize=12
        )
        fig.tight_layout()

        frame_path = tmp_dir / f"frame_N{N:04d}.png"
        fig.savefig(frame_path, dpi=100)  # no bbox_inches so all frames same size
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"[GIF] saved  {gif_path}  ({len(frames)} frames)")


def make_rotation_gif(
    N: int,
    U: np.ndarray,
    gif_path: str | Path,
    n_frames: int = 36,
    fps: int = 12,
) -> None:
    """360° rotation GIF of the 3D surface plot."""
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    nodes, h = build_grid(N)
    x, y = node_coords(nodes, h)
    xi, yi = np.unique(x), np.unique(y)
    Z = U.reshape(N, N)
    XX, YY = np.meshgrid(xi, yi)

    tmp_dir = gif_path.parent / "_rot_frames"
    tmp_dir.mkdir(exist_ok=True)
    frames = []

    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(XX, YY, Z, cmap=CMAP, linewidth=0, antialiased=True)
        ax.view_init(elev=30, azim=angle)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_title(f"u(x,y) – N={N}")
        fig.tight_layout()

        fp = tmp_dir / f"rot_{i:03d}.png"
        fig.savefig(fp, dpi=90)  # consistent size, no bbox_inches
        plt.close(fig)
        frames.append(imageio.imread(fp))

    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"[GIF] saved  {gif_path}  ({n_frames} frames, 360° rotation)")
