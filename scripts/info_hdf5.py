"""scripts/info_hdf5.py — print HDF5 file structure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import h5py

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/hdf5/poisson_sin.hdf5"

print(f"\n  File: {path}")
print(f"  {'─'*52}")

with h5py.File(path, "r") as fh:
    def show(name, obj):
        indent = "  " + "  " * name.count("/")
        if isinstance(obj, h5py.Group):
            attrs = {k: f"{v:.4e}" if isinstance(v, float) else v for k, v in obj.attrs.items()}
            print(f"{indent}📁  {name}/    {attrs}")
        else:
            print(f"{indent}📄  {name}    shape={obj.shape}  dtype={obj.dtype}")

    fh.visititems(show)
    if fh.attrs:
        print(f"\n  root attrs: { {k: round(v,4) if isinstance(v,float) else v for k,v in fh.attrs.items()} }")

print()
