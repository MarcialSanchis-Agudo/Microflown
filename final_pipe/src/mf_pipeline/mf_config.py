"""Centralized configuration for paths used across the pipeline."""
from pathlib import Path
import os

# Repository root = final_pipe/
ROOT = Path(__file__).resolve().parents[2]

# Output directory (PNG/NPZ/CSV). Override with MF_EXPORTS.
EXPORTS = Path(os.environ.get("MF_EXPORTS", ROOT / "exports"))
EXPORTS.mkdir(parents=True, exist_ok=True)

# Measurement H5 (ParaView export). Override with MF_H5_PATH/MF_H5_GROUP.
DEFAULT_H5 = ROOT.parent / "monopole_paraview_data" / "low frequency monopole data - 4096 NFFT - 25 mm.h5"
H5_PATH = Path(os.environ.get("MF_H5_PATH", DEFAULT_H5))
H5_GROUP = os.environ.get("MF_H5_GROUP", "/Proc-2_3D_Cuboid,20_mm_resolution")
