"""
vtk_io.py
---------
Write Poisson-solution data to legacy VTK ASCII format for Paraview.

Two export modes:
  * ``mode='2d'`` – scalar FIELD data on the z=0 plane
  * ``mode='3d'`` – VECTORS data with z-component = u  (creates a height field)

Author  : M1 CHPS – UPVD 2025-2026
Supervisor: Serge Dumont  <https://perso.univ-perp.fr/sdumont/>
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from solver import build_grid, node_coords, build_laplacian, build_rhs, solve_direct


# ---------------------------------------------------------------------------
# Low-level VTK text builders
# ---------------------------------------------------------------------------

def _header(title: str) -> str:
    return f"# vtk DataFile Version 3.0\n{title}\nASCII\n\n"


def _points(N: int, h: float, mode: Literal["2d", "3d"]) -> str:
    M = N + 2           # total grid size including boundary
    n_pts = M * M * (2 if mode == "3d" else 1)
    lines = [f"DATASET UNSTRUCTURED_GRID\nPOINTS {n_pts} float"]

    z_values = [0.0, 0.01] if mode == "3d" else [0.0]
    for z in z_values:
        for i in range(M):
            for j in range(M):
                lines.append(f"{i*h:.6f} {j*h:.6f} {z:.2f}")

    return "\n".join(lines) + "\n\n"


def _cells(N: int, mode: Literal["2d", "3d"]) -> str:
    M = N + 2
    n_cells = (M - 1) ** 2
    pts_per_cell = 8 if mode == "3d" else 4
    total_ints = n_cells * (pts_per_cell + 1)

    lines = [f"CELLS {n_cells} {total_ints}"]
    offset2 = M * M  # second layer offset (3D)

    for i in range(M - 1):
        for j in range(M - 1):
            k = i * M + j
            if mode == "2d":
                lines.append(f"4 {k} {k+1} {k+M+1} {k+M}")
            else:
                k2 = offset2 + k
                lines.append(
                    f"8 {k} {k+1} {k+M+1} {k+M} {k2} {k2+1} {k2+M+1} {k2+M}"
                )

    return "\n".join(lines) + "\n\n"


def _cell_types(N: int, mode: Literal["2d", "3d"]) -> str:
    M = N + 2
    n_cells = (M - 1) ** 2
    cell_type = 12 if mode == "3d" else 9
    lines = [f"CELL_TYPES {n_cells}"]
    lines += [str(cell_type)] * n_cells
    return "\n".join(lines) + "\n\n"


def _point_data_header(N: int, mode: Literal["2d", "3d"], n_fields: int) -> str:
    M = N + 2
    n_pts = M * M * (2 if mode == "3d" else 1)
    out = f"POINT_DATA {n_pts}\n"
    if mode == "2d":
        out += f"FIELD FieldData {n_fields}\n"
    return out


def _scalar_field(name: str, N: int, U: np.ndarray) -> str:
    M = N + 2
    n_pts = M * M
    lines = [f"{name} 1 {n_pts} float"]
    k = 0
    for i in range(M):
        for j in range(M):
            if i == 0 or i == M - 1 or j == 0 or j == M - 1:
                lines.append("0.000000")
            else:
                lines.append(f"{U[k]:.6f}")
                k += 1
    return "\n".join(lines) + "\n"


def _vector_field(name: str, N: int, U: np.ndarray, layer: int = 1) -> str:
    """Write z-component = U, x=y=0 (creates a height-map in Paraview)."""
    M = N + 2
    lines = [f"\nVECTORS {name} float"]
    for _ in range(layer):
        k = 0
        for i in range(M):
            for j in range(M):
                if i == 0 or i == M - 1 or j == 0 or j == M - 1:
                    lines.append("0.0 0.0 0.0")
                else:
                    lines.append(f"0.0 0.0 {U[k]:.6f}")
                    k += 1
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_vtk(
    path: str | Path,
    N: int,
    fields: dict[str, np.ndarray],
    title: str = "Poisson2D",
    mode: Literal["2d", "3d"] = "2d",
) -> None:
    """Write one VTK file.

    Parameters
    ----------
    path   : output file path (.vtk)
    N      : number of interior nodes per direction
    fields : dict mapping field name -> 1D numpy array of length N²
    title  : dataset title embedded in the file header
    mode   : '2d' (scalar) or '3d' (vector height map)
    """
    _, h = build_grid(N)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = _header(title)
    content += _points(N, h, mode)
    content += _cells(N, mode)
    content += _cell_types(N, mode)
    content += _point_data_header(N, mode, len(fields))

    for name, U in fields.items():
        if mode == "2d":
            content += _scalar_field(name, N, U)
        else:
            content += _vector_field(name, N, U, layer=2)

    path.write_text(content)
    print(f"[VTK] wrote {path}  (N={N}, mode={mode})")
