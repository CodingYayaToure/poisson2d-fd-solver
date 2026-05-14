"""
solver.py
---------
Finite-difference solver for the 2D Poisson problem on the unit square.

    -Δu = f   in Ω = (0,1)²
     u  = 0   on ∂Ω

Discretisation: 5-point finite-difference stencil on a uniform N×N interior grid.
Matrix assembly uses CSR sparse format for memory efficiency.

Author  : M1 CHPS – UPVD 2025-2026
Supervisor: Serge Dumont  <https://perso.univ-perp.fr/sdumont/>
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def build_grid(N: int) -> tuple[np.ndarray, float]:
    """Return (nodes, h) for an N×N interior grid.

    Numbering: k = (i-1) + N*(j-1),  1 ≤ i,j ≤ N
    nodes[k] = [i, j]
    """
    h = 1.0 / (N + 1)
    idx = np.arange(N * N)
    i_idx = idx % N + 1          # column index 1..N
    j_idx = idx // N + 1         # row index    1..N
    nodes = np.column_stack([i_idx, j_idx])
    return nodes, h


def node_coords(nodes: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Physical (x,y) coordinates for every interior node."""
    return nodes[:, 0] * h, nodes[:, 1] * h


# ---------------------------------------------------------------------------
# Sparse Laplacian matrix
# ---------------------------------------------------------------------------

def build_laplacian(N: int, h: float) -> sp.csr_matrix:
    """Assemble the N²×N² sparse finite-difference Laplacian matrix.

    Uses scipy.sparse.diags for efficient CSR construction instead of
    building row by row.
    """
    n2 = N * N
    inv_h2 = 1.0 / h**2

    # Main diagonal: 4/h²
    diag_main = np.full(n2, 4.0 * inv_h2)

    # ±1 off-diagonal (horizontal neighbours), skip across block boundary
    diag_h = np.full(n2 - 1, -inv_h2)
    diag_h[np.arange(N - 1, n2 - 1, N)] = 0.0   # zero at block boundaries

    # ±N off-diagonal (vertical neighbours)
    diag_v = np.full(n2 - N, -inv_h2)

    A = sp.diags(
        [diag_main, diag_h, diag_h, diag_v, diag_v],
        offsets=[0, 1, -1, N, -N],
        shape=(n2, n2),
        format="csr",
    )
    return A


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------

def build_rhs(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    nodes: np.ndarray,
    h: float,
) -> np.ndarray:
    """Evaluate f at all interior nodes and return the RHS vector B."""
    x, y = node_coords(nodes, h)
    return f(x, y)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_direct(A: sp.csr_matrix, B: np.ndarray) -> np.ndarray:
    """Sparse LU factorisation (SuperLU via scipy)."""
    return spla.spsolve(A, B)


def solve_dense_lu(A: sp.csr_matrix, B: np.ndarray) -> np.ndarray:
    """Dense LU (scipy.linalg) — kept for small N / benchmarking."""
    A_dense = A.toarray()
    lu, piv = la.lu_factor(A_dense)
    return la.lu_solve((lu, piv), B)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def l2_norm(v: np.ndarray, h: float) -> float:
    """Discrete L² norm: h × ‖v‖_ℓ²."""
    return h * float(np.linalg.norm(v))


def convergence_study(
    N_values: list[int],
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    u_exact: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    """Run the solver for each N in *N_values* and collect metrics.

    Returns a dict with keys: 'N', 'h', 'error_l2', 'cpu_time'.
    """
    results: dict[str, list] = {"N": [], "h": [], "error_l2": [], "cpu_time": []}

    for N in N_values:
        nodes, h = build_grid(N)
        x, y = node_coords(nodes, h)

        t0 = time.perf_counter()
        A = build_laplacian(N, h)
        B = build_rhs(f, nodes, h)
        U = solve_direct(A, B)
        cpu = time.perf_counter() - t0

        Uex = u_exact(x, y)
        err = l2_norm(U - Uex, h)

        results["N"].append(N)
        results["h"].append(h)
        results["error_l2"].append(err)
        results["cpu_time"].append(cpu)

    return {k: np.array(v) for k, v in results.items()}


def estimate_convergence_rate(h_arr: np.ndarray, err_arr: np.ndarray) -> tuple[float, float]:
    """Fit log(err) = α·log(h) + log(C) by least-squares.

    Returns (α, C).
    """
    C_mat = np.column_stack([np.log(h_arr), np.ones(len(h_arr))])
    coeffs, *_ = np.linalg.lstsq(C_mat, np.log(err_arr), rcond=None)
    return float(coeffs[0]), float(np.exp(coeffs[1]))
