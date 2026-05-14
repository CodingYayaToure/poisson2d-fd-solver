"""
app.py
------
Interactive web dashboard for the 2D Poisson FD solver, built with Panel.

Features
--------
* Solve interactively for any N using the sin/sin or f=1 test case
* Upload a custom VTK file and view the scalar fields it contains
* Side-by-side plots: numerical, exact (if available), and point-wise error
* Convergence study widget

Run
---
    pip install panel bokeh
    python src/app.py

Author  : M1 CHPS – UPVD 2025-2026
Supervisor: Serge Dumont  <https://perso.univ-perp.fr/sdumont/>
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

import panel as pn
pn.extension()

from .solver import (
    build_grid,
    build_laplacian,
    build_rhs,
    node_coords,
    solve_direct,
    l2_norm,
    convergence_study,
    estimate_convergence_rate,
)

# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

CASES = {
    "sin(2πx)sin(2πy)": {
        "f": lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y),
        "u_exact": lambda x, y: (1 / (8 * np.pi**2))
        * np.sin(2 * np.pi * x)
        * np.sin(2 * np.pi * y),
        "exact_known": True,
    },
    "f = 1 (uniform source)": {
        "f": lambda x, y: np.ones_like(x),
        "u_exact": None,
        "exact_known": False,
    },
}

CMAP = "RdYlBu_r"

# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

title = pn.pane.Markdown(
    """
# 🔬 2D Poisson Solver — Interactive Visualisation
**M1 CHPS · UPVD 2025-2026 · Encadré par [Serge Dumont](https://perso.univ-perp.fr/sdumont/)**

---
""",
    width=820,
)

case_select = pn.widgets.Select(
    name="Test case", options=list(CASES.keys()), value=list(CASES.keys())[0], width=260
)
N_slider = pn.widgets.IntSlider(
    name="N (interior nodes per direction)", start=5, end=150, step=5, value=30, width=400
)
solve_btn = pn.widgets.Button(name="▶  Solve", button_type="primary", width=140)
status_md = pn.pane.Markdown("", width=600)

# Convergence study
N_min_input = pn.widgets.IntInput(name="N min", value=5, step=5, width=120)
N_max_input = pn.widgets.IntInput(name="N max", value=100, step=10, width=120)
N_steps_input = pn.widgets.IntInput(name="Steps", value=8, width=100)
conv_btn = pn.widgets.Button(name="▶  Run convergence study", button_type="success", width=220)

# VTK file upload
vtk_upload = pn.widgets.FileInput(accept=".vtk", name="Upload VTK file")
vtk_field_select = pn.widgets.Select(name="Field to display", options=[], width=200)
vtk_load_btn = pn.widgets.Button(name="Load VTK", button_type="warning", width=140)
vtk_status = pn.pane.Markdown("", width=600)

# Plot panes
plot_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", height=420)
conv_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", height=420)
vtk_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", height=420)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_pane(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return pn.pane.PNG(buf.read(), sizing_mode="stretch_width")


def _make_solution_fig(N, U, Uex, err, h, case_name):
    has_exact = Uex is not None
    ncols = 3 if has_exact else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    nodes, _ = build_grid(N)
    x, y = node_coords(nodes, h)
    xi, yi = np.unique(x), np.unique(y)

    datasets = [U]
    labels = [f"Numerical U  (N={N})"]
    if has_exact:
        datasets += [Uex, np.abs(U - Uex)]
        labels += ["Exact Uex", f"|U − Uex|  max={np.abs(U-Uex).max():.2e}"]

    for ax, data, lbl in zip(axes, datasets, labels):
        cf = ax.contourf(xi, yi, data.reshape(N, N), levels=40, cmap=CMAP)
        fig.colorbar(cf, ax=ax, format="%.2e")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.suptitle(f"Case: {case_name}  |  h = {h:.4f}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

result_store: dict = {}


def on_solve(event):
    status_md.object = "⏳ Solving…"
    case = CASES[case_select.value]
    N = N_slider.value

    nodes, h = build_grid(N)
    x, y = node_coords(nodes, h)
    A = build_laplacian(N, h)
    B = build_rhs(case["f"], nodes, h)

    t0 = time.perf_counter()
    U = solve_direct(A, B)
    cpu = time.perf_counter() - t0

    Uex = case["u_exact"](x, y) if case["exact_known"] else None
    err_val = l2_norm(U - Uex, h) if Uex is not None else float("nan")

    result_store.update({"N": N, "h": h, "U": U, "Uex": Uex, "x": x, "y": y})

    fig = _make_solution_fig(N, U, Uex, None, h, case_select.value)
    plot_pane.object = fig

    msg = f"✔ Solved in {cpu*1e3:.1f} ms  |  N={N}  |  h={h:.4f}"
    if np.isfinite(err_val):
        msg += f"  |  ‖error‖_L² = {err_val:.4e}"
    status_md.object = msg


def on_convergence(event):
    case = CASES[case_select.value]
    N_min, N_max, steps = N_min_input.value, N_max_input.value, N_steps_input.value
    N_values = np.linspace(N_min, N_max, steps, dtype=int).tolist()

    status_md.object = f"⏳ Running convergence study for N ∈ {N_values}…"

    if not case["exact_known"]:
        status_md.object = "⚠ Exact solution not available for this test case."
        return

    metrics = convergence_study(N_values, case["f"], case["u_exact"])
    alpha, C = estimate_convergence_rate(metrics["h"], metrics["error_l2"])

    h_fit = np.linspace(metrics["h"].min(), metrics["h"].max(), 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Error plot
    axes[0].loglog(metrics["h"], metrics["error_l2"], "o-", ms=6, label="‖error‖_L²")
    axes[0].loglog(
        h_fit,
        C * h_fit**alpha,
        "--r",
        label=rf"Fit  $\alpha \approx {alpha:.2f}$",
    )
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("L² error")
    axes[0].set_title("Convergence (log-log)")
    axes[0].legend()
    axes[0].grid(True, which="both", ls=":", alpha=0.5)

    # CPU time
    axes[1].loglog(metrics["N"], metrics["cpu_time"], "s-g", ms=6)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("CPU time (s)")
    axes[1].set_title("CPU time vs N")
    axes[1].grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle(f"Convergence study – {case_select.value}", fontsize=11)
    fig.tight_layout()
    conv_pane.object = fig
    status_md.object = f"✔ Convergence rate  α ≈ {alpha:.3f}  |  C ≈ {C:.4e}"


def _parse_vtk_scalars(content: str) -> dict[str, np.ndarray]:
    """Minimal VTK ASCII scalar parser – handles FIELD and SCALARS sections."""
    fields: dict[str, np.ndarray] = {}
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # FIELD FieldData n
        if line.startswith("FIELD"):
            parts = line.split()
            n_fields = int(parts[2])
            i += 1
            for _ in range(n_fields):
                while i < len(lines) and not lines[i].strip():
                    i += 1
                fline = lines[i].strip().split()
                fname, _, n_vals = fline[0], fline[1], int(fline[2])
                i += 1
                vals = []
                while len(vals) < n_vals and i < len(lines):
                    vals.extend(map(float, lines[i].split()))
                    i += 1
                fields[fname] = np.array(vals)
            continue

        # SCALARS name type [ncomp]
        if line.startswith("SCALARS"):
            fname = line.split()[1]
            i += 1
            # skip LOOKUP_TABLE line
            while i < len(lines) and lines[i].strip().startswith("LOOKUP_TABLE"):
                i += 1
            vals = []
            while i < len(lines) and lines[i].strip() and not lines[i][0].isalpha():
                vals.extend(map(float, lines[i].split()))
                i += 1
            fields[fname] = np.array(vals)
            continue

        i += 1
    return fields


vtk_data_store: dict = {}


def on_vtk_upload(event):
    if vtk_upload.value is None:
        return
    try:
        raw = vtk_upload.value
        if isinstance(raw, bytes):
            content = raw.decode("utf-8", errors="replace")
        else:
            content = raw
        fields = _parse_vtk_scalars(content)
        vtk_data_store["fields"] = fields
        vtk_field_select.options = list(fields.keys()) if fields else ["(none)"]
        vtk_status.object = f"✔ Loaded `{vtk_upload.filename}` — fields: {list(fields.keys())}"
    except Exception as exc:
        vtk_status.object = f"❌ Parse error: {exc}"


def on_vtk_load(event):
    fields = vtk_data_store.get("fields", {})
    fname = vtk_field_select.value
    if fname not in fields:
        vtk_status.object = "⚠ No field selected."
        return
    arr = fields[fname]
    # Guess N: total_pts = (N+2)² or 2*(N+2)²
    n_pts = len(arr)
    # Try to get N from points count
    for mode_factor in [1, 2]:
        total = n_pts // mode_factor
        Np2 = int(round(total**0.5))
        if Np2**2 == total:
            N_vtk = Np2 - 2
            break
    else:
        vtk_status.object = "⚠ Cannot infer grid size from point count."
        return

    # Extract interior points (skip boundary rows/cols)
    M = N_vtk + 2
    flat = arr[:M * M] if len(arr) > M * M else arr
    grid = flat.reshape(M, M)
    interior = grid[1:-1, 1:-1]

    xi = np.linspace(0, 1, N_vtk)
    yi = np.linspace(0, 1, N_vtk)

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(xi, yi, interior, levels=40, cmap=CMAP)
    fig.colorbar(cf, ax=ax, label=fname)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"VTK field: {fname}  (N={N_vtk})")
    ax.set_aspect("equal")
    fig.tight_layout()

    vtk_pane.object = fig
    vtk_status.object = f"✔ Displaying field `{fname}`  (N={N_vtk})"


solve_btn.on_click(on_solve)
conv_btn.on_click(on_convergence)
vtk_upload.param.watch(on_vtk_upload, "value")
vtk_load_btn.on_click(on_vtk_load)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

solver_tab = pn.Column(
    pn.pane.Markdown("### Solve & Visualise"),
    pn.Row(case_select, N_slider, solve_btn),
    status_md,
    plot_pane,
)

convergence_tab = pn.Column(
    pn.pane.Markdown("### Convergence Study"),
    pn.Row(N_min_input, N_max_input, N_steps_input, conv_btn),
    conv_pane,
)

vtk_tab = pn.Column(
    pn.pane.Markdown("### Upload & Inspect a VTK file"),
    pn.pane.Markdown(
        "Load any VTK file produced by the solver (or your own) to visualise the scalar fields."
    ),
    pn.Row(vtk_upload, vtk_field_select, vtk_load_btn),
    vtk_status,
    vtk_pane,
)

dashboard = pn.Column(
    title,
    pn.Tabs(
        ("🔍 Solve", solver_tab),
        ("📈 Convergence", convergence_tab),
        ("📂 VTK viewer", vtk_tab),
    ),
    sizing_mode="stretch_width",
    margin=(10, 20),
)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pn.serve(dashboard, port=5006, show=True, title="Poisson 2D – CHPS UPVD")
