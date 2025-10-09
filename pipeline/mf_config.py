# mf_pipeline/mf_config.py
from pathlib import Path

# ABSOLUTE output directory (all PNG/NPZ/CSV land here)
EXPORTS = Path("/Users/marcialsanchis/Desktop/PhD/ModeAir/MicroFLown/mf_pipeline/exports")
EXPORTS.mkdir(parents=True, exist_ok=True)

# H5 location (absolute)
H5_PATH = Path("/Users/marcialsanchis/Desktop/PhD/ModeAir/MicroFLown/exports") / "Loudspeaker_20mm.h5"
H5_GROUP = "/Proc-2_3D_Cuboid,20_mm_resolution"
