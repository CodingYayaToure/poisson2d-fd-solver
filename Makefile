# =============================================================================
#  Makefile — 2D Poisson FD Solver
#  M1 CHPS · UPVD 2025-2026
#  Encadré par Serge Dumont  https://perso.univ-perp.fr/sdumont/
# =============================================================================

PYTHON      := python3
SRC         := src
OUT         := outputs
VTK_DIR     := $(OUT)/vtk
HDF5_DIR    := $(OUT)/hdf5
FIG_DIR     := $(OUT)/figures
GIF_DIR     := $(OUT)/gif

# Default VTK file opened by Paraview
VTK_DEFAULT := $(VTK_DIR)/poisson_sin_2d.vtk

# Paraview executable (override on command line: make paraview PARAVIEW=/path/to/pvpython)
PARAVIEW    := paraview
PVPYTHON    := pvpython

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
#  Help
# -----------------------------------------------------------------------------

.PHONY: help
help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║  2D Poisson FD Solver — Makefile                        ║"
	@echo "  ║  M1 CHPS · UPVD 2025-2026                               ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Usage:  make <target>"
	@echo ""
	@echo "  Setup"
	@echo "  ─────"
	@echo "  install          Install Python dependencies (pip)"
	@echo "  venv             Create + activate a virtual environment"
	@echo ""
	@echo "  Pipeline"
	@echo "  ────────"
	@echo "  run              Run full pipeline (VTK + HDF5 + figures + GIFs)"
	@echo "  solve            Solver + convergence study only"
	@echo "  vtk              Generate VTK files only"
	@echo "  hdf5             Generate HDF5 files only"
	@echo "  figures          Generate static PNG figures only"
	@echo "  gif              Generate animated GIFs only"
	@echo ""
	@echo "  Visualisation"
	@echo "  ─────────────"
	@echo "  paraview         Open default VTK in Paraview (GUI)"
	@echo "  paraview-sin2d   Open sin/sin solution (2D scalar)"
	@echo "  paraview-sin3d   Open sin/sin solution (3D height map)"
	@echo "  paraview-const2d Open f=1 solution (2D scalar)"
	@echo "  paraview-const3d Open f=1 solution (3D height map)"
	@echo "  dashboard        Launch Panel interactive dashboard (browser)"
	@echo ""
	@echo "  Utilities"
	@echo "  ─────────"
	@echo "  info-vtk         Print summary of all generated VTK files"
	@echo "  info-hdf5        Print HDF5 file structure"
	@echo "  clean            Remove all generated outputs"
	@echo "  clean-cache      Remove Python __pycache__ only"
	@echo ""
	@echo "  Override Paraview path:"
	@echo "  make paraview PARAVIEW=/usr/bin/paraview"
	@echo ""

# -----------------------------------------------------------------------------
#  Setup
# -----------------------------------------------------------------------------

.PHONY: install
install:
	@echo "→ Installing dependencies…"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✔ Done."

.PHONY: venv
venv:
	@echo "→ Creating virtual environment in .venv/"
	$(PYTHON) -m venv .venv
	@echo "✔ Run:  source .venv/bin/activate"

# -----------------------------------------------------------------------------
#  Output directories
# -----------------------------------------------------------------------------

$(VTK_DIR) $(HDF5_DIR) $(FIG_DIR) $(GIF_DIR):
	mkdir -p $@

# -----------------------------------------------------------------------------
#  Pipeline targets
# -----------------------------------------------------------------------------

.PHONY: run
run: $(VTK_DIR) $(HDF5_DIR) $(FIG_DIR) $(GIF_DIR)
	@echo "→ Running full pipeline…"
	$(PYTHON) $(SRC)/main.py
	@echo ""
	@echo "✔ Outputs written to $(OUT)/"
	@echo "   VTK    → $(VTK_DIR)/"
	@echo "   HDF5   → $(HDF5_DIR)/"
	@echo "   Figures → $(FIG_DIR)/"
	@echo "   GIFs   → $(GIF_DIR)/"

# Individual sub-targets calling main.py with a mode flag
# (can also be driven separately if you split main.py)

.PHONY: solve
solve: $(FIG_DIR)
	@echo "→ Convergence study + figures…"
	$(PYTHON) -c "\
import sys; sys.path.insert(0,'$(SRC)'); \
import numpy as np; \
from solver import convergence_study, estimate_convergence_rate; \
from plotting import plot_convergence, plot_cpu_time; \
f    = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y); \
uex  = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y)/(8*np.pi**2); \
m = convergence_study([4,9,19,49,99,199], f, uex); \
a,C = estimate_convergence_rate(m['h'],m['error_l2']); \
print(f'  α ≈ {a:.4f}   C ≈ {C:.4e}'); \
plot_convergence(m['h'],m['error_l2'],save_path='$(FIG_DIR)/convergence.png'); \
plot_cpu_time(m['N'],m['cpu_time'],save_path='$(FIG_DIR)/cpu_time.png'); \
print('✔ Figures saved.')"

.PHONY: vtk
vtk: $(VTK_DIR)
	@echo "→ Generating VTK files…"
	$(PYTHON) -c "\
import sys, numpy as np; sys.path.insert(0,'$(SRC)'); \
from solver import build_grid, build_laplacian, build_rhs, node_coords, solve_direct; \
from vtk_io import write_vtk; \
f   = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y); \
uex = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y)/(8*np.pi**2); \
N=99; nd,h=build_grid(N); x,y=node_coords(nd,h); \
U=solve_direct(build_laplacian(N,h), build_rhs(f,nd,h)); Uex=uex(x,y); \
write_vtk('$(VTK_DIR)/poisson_sin_2d.vtk',   N, {'U':U,'Uex':Uex}, 'Poisson2D sin/sin', mode='2d'); \
write_vtk('$(VTK_DIR)/poisson_sin_3d.vtk',   N, {'U':U,'Uex':Uex}, 'Poisson2D sin/sin', mode='3d'); \
f2=lambda x,y: np.ones_like(x); U2=solve_direct(build_laplacian(N,h), build_rhs(f2,nd,h)); \
write_vtk('$(VTK_DIR)/poisson_const_2d.vtk', N, {'U':U2}, 'Poisson2D f=1', mode='2d'); \
write_vtk('$(VTK_DIR)/poisson_const_3d.vtk', N, {'U':U2}, 'Poisson2D f=1', mode='3d')"

.PHONY: hdf5
hdf5: $(HDF5_DIR)
	@echo "→ Generating HDF5 file…"
	$(PYTHON) -c "\
import sys, numpy as np; sys.path.insert(0,'$(SRC)'); \
from hdf5_io import write_hdf5; \
f   = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y); \
uex = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y)/(8*np.pi**2); \
write_hdf5('$(HDF5_DIR)/poisson_sin.hdf5', [4,9,19,49,99,199], f, uex)"

.PHONY: figures
figures: $(FIG_DIR)
	@echo "→ Generating static figures…"
	$(PYTHON) -c "\
import sys, numpy as np; sys.path.insert(0,'$(SRC)'); \
from solver import build_grid, build_laplacian, build_rhs, node_coords, solve_direct, convergence_study, estimate_convergence_rate; \
from plotting import plot_solution, plot_solution_3d, plot_error_and_exact, plot_convergence, plot_cpu_time; \
f   = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y); \
uex = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y)/(8*np.pi**2); \
N=99; nd,h=build_grid(N); x,y=node_coords(nd,h); \
U=solve_direct(build_laplacian(N,h), build_rhs(f,nd,h)); Uex=uex(x,y); \
plot_solution(N,U,save_path='$(FIG_DIR)/solution_2d.png'); \
plot_solution_3d(N,U,save_path='$(FIG_DIR)/solution_3d.png'); \
plot_error_and_exact(N,U,Uex,save_path='$(FIG_DIR)/solution_comparison.png'); \
m=convergence_study([4,9,19,49,99,199],f,uex); \
plot_convergence(m['h'],m['error_l2'],save_path='$(FIG_DIR)/convergence.png'); \
plot_cpu_time(m['N'],m['cpu_time'],save_path='$(FIG_DIR)/cpu_time.png'); \
print('✔ All figures saved.')"

.PHONY: gif
gif: $(GIF_DIR)
	@echo "→ Generating animated GIFs…"
	$(PYTHON) -c "\
import sys, numpy as np; sys.path.insert(0,'$(SRC)'); \
from solver import build_grid, build_laplacian, build_rhs, node_coords, solve_direct; \
from plotting import make_convergence_gif, make_rotation_gif; \
f   = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y); \
uex = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y)/(8*np.pi**2); \
make_convergence_gif([4,9,19,39,59,79,99],f,uex,gif_path='$(GIF_DIR)/convergence.gif',fps=1); \
N=99; nd,h=build_grid(N); U=solve_direct(build_laplacian(N,h),build_rhs(f,nd,h)); \
make_rotation_gif(N,U,gif_path='$(GIF_DIR)/rotation_3d.gif',n_frames=36,fps=12); \
print('✔ GIFs saved.')"

# -----------------------------------------------------------------------------
#  Paraview targets
# -----------------------------------------------------------------------------

.PHONY: paraview
paraview: $(VTK_DEFAULT)
	@echo "→ Opening $(VTK_DEFAULT) in Paraview…"
	$(PARAVIEW) $(VTK_DEFAULT) &

.PHONY: paraview-sin2d
paraview-sin2d: $(VTK_DIR)/poisson_sin_2d.vtk
	@echo "→ Opening sin/sin 2D solution in Paraview…"
	$(PARAVIEW) $(VTK_DIR)/poisson_sin_2d.vtk &

.PHONY: paraview-sin3d
paraview-sin3d: $(VTK_DIR)/poisson_sin_3d.vtk
	@echo "→ Opening sin/sin 3D height-map in Paraview…"
	$(PARAVIEW) $(VTK_DIR)/poisson_sin_3d.vtk &

.PHONY: paraview-const2d
paraview-const2d: $(VTK_DIR)/poisson_const_2d.vtk
	@echo "→ Opening f=1 2D solution in Paraview…"
	$(PARAVIEW) $(VTK_DIR)/poisson_const_2d.vtk &

.PHONY: paraview-const3d
paraview-const3d: $(VTK_DIR)/poisson_const_3d.vtk
	@echo "→ Opening f=1 3D height-map in Paraview…"
	$(PARAVIEW) $(VTK_DIR)/poisson_const_3d.vtk &

# Batch render via pvpython (headless, produces a screenshot)
.PHONY: paraview-render
paraview-render: $(VTK_DIR)/poisson_sin_2d.vtk $(FIG_DIR)
	@echo "→ Batch rendering VTK via pvpython…"
	$(PVPYTHON) scripts/paraview_render.py \
		--input  $(VTK_DIR)/poisson_sin_2d.vtk \
		--output $(FIG_DIR)/paraview_render.png \
		--field  U
	@echo "✔ Render saved to $(FIG_DIR)/paraview_render.png"

# -----------------------------------------------------------------------------
#  Dashboard
# -----------------------------------------------------------------------------

.PHONY: dashboard
dashboard:
	@echo "→ Starting Panel dashboard on http://localhost:5006"
	@echo "   Press Ctrl-C to stop."
	$(PYTHON) $(SRC)/app.py

# -----------------------------------------------------------------------------
#  Info / inspection utilities
# -----------------------------------------------------------------------------

.PHONY: info-vtk
info-vtk:
	@echo ""
	@echo "  VTK files in $(VTK_DIR)/"
	@echo "  ─────────────────────────────────────────────────────"
	@for f in $(VTK_DIR)/*.vtk; do \
		echo ""; \
		echo "  ► $$f"; \
		head -4 $$f | sed 's/^/    /'; \
		echo "    Size: $$(du -h $$f | cut -f1)"; \
		echo "    Points: $$(grep -m1 '^POINTS' $$f || echo 'n/a')"; \
		echo "    Fields: $$(grep -E '^[A-Z]+ [A-Za-z]+ 1 [0-9]+' $$f | awk '{print $$1}' | tr '\n' ' ')"; \
	done
	@echo ""

.PHONY: info-hdf5
info-hdf5:
	@echo "→ HDF5 structure:"
	$(PYTHON) scripts/info_hdf5.py $(HDF5_DIR)/poisson_sin.hdf5

# -----------------------------------------------------------------------------
#  Clean
# -----------------------------------------------------------------------------

.PHONY: clean
clean:
	@echo "→ Removing all generated outputs…"
	rm -rf $(OUT)
	@echo "✔ $(OUT)/ removed."

.PHONY: clean-cache
clean-cache:
	@echo "→ Removing Python cache…"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✔ Cache cleared."

.PHONY: clean-gif
clean-gif:
	rm -rf $(GIF_DIR)/_frames $(GIF_DIR)/_rot_frames
	@echo "✔ Temporary GIF frame directories removed."

# -----------------------------------------------------------------------------
#  Phony declarations
# -----------------------------------------------------------------------------

.PHONY: all
all: run
