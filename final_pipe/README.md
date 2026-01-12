# Microflown Pipeline (src layout)

Cleaned-up structure for the Microflown near→far pipeline with a `src/` package, scripts, and bundled example data you already ran.

## Layout
- `src/mf_pipeline/` core modules (fill, loader, directivity, plots, config).
- `scripts/` CLI helpers (Python + bash wrappers). They assume `PYTHONPATH=src` or `pip install -e .`.
- `figs/paper_figs/` precomputed NPZ + PNGs from your previous runs (near25 distances/frequencies).
- `exports/` default output drop point (override with `MF_OUTDIR` or `MF_EXPORTS`).
- `pyproject.toml` minimal metadata for editable installs.

## Install
```bash
python -m pip install -e .
# or just export PYTHONPATH for ad-hoc runs
export PYTHONPATH=src
```

## Config (paths)
- `MF_H5_PATH` (default: `../monopole_paraview_data/low frequency monopole data - 4096 NFFT - 25 mm.h5`)
- `MF_H5_GROUP` (default: `/Proc-2_3D_Cuboid,20_mm_resolution`)
- `MF_EXPORTS` / `MF_OUTDIR` to change where NPZ/PNGs land (default: `exports/`).

## Practical examples (using existing data)
All commands from repo root (`final_pipe/`):

1) Far-field from an existing filled array (near25, 1000 Hz):
```bash
python scripts/run_directivity.py \
  --npz figs/paper_figs/filled_arrays_near25_1000Hz.npz \
  --rfar 1.0 --auto-center --db --tag near25_demo
# Outputs -> exports/filled_arrays_near25_demo_1000Hz_p_map_lonlat.npz + polar PNGs
```

2) Build a directionality sheet from your precomputed maps at multiple radii (1000 Hz):
```bash
python scripts/run_directionality_sheet.py \
  --maps figs/paper_figs/filled_arrays_near25_at_0p5m_1000Hz_p_map_lonlat.npz \
         figs/paper_figs/filled_arrays_near25_at_1p0m_1000Hz_p_map_lonlat.npz \
         figs/paper_figs/filled_arrays_near25_at_5p0m_1000Hz_p_map_lonlat.npz \
  --labels "0.5 m" "1.0 m" "5.0 m" \
  --out exports/directionality_near25_1000Hz_radii.png
```

3) Quick near-field visuals on your existing runs:
- `bash scripts/extra_near.sh` (plots magnitude + slices for near25 1008/4000 Hz).
- Or directly: `python scripts/run_plot_source.py --npz figs/paper_figs/filled_arrays_near25_1008Hz.npz --metric u_mag`

4) CSM propagation (if you have a CSM + transfer matrix):
```bash
# G_npz must contain G (and optionally dGdx/dGdy/dGdz); csm NPZ must contain Sqq
python scripts/run_csm_propagate.py \
  --G-npz path/to/G_and_gradients.npz \
  --csm path/to/source_csm.npz \
  --f0 1000 --tag demo_csm
# Outputs: Spp NPZ, SPL plot, optional intensity CSV if gradients provided.
```

5) Sanity checks on data:
```bash
python scripts/run_inspect_h5.py --h5 "../monopole_paraview_data/low frequency monopole data - 4096 NFFT - 25 mm.h5" --group "Proc-3_3D_Cuboid,25_mm_resolution"
python scripts/run_inspect_npz.py --npz figs/examples/filled_arrays_near25_h5_1000Hz.npz
python scripts/run_imag_hist.py --npz figs/examples/filled_arrays_near25_h5_1000Hz.npz
```

## Notes
- All scripts now resolve the package via `src/`; no more absolute paths. Override config with env vars if needed.
- Default outputs stay in `exports/` to keep the bundled `figs/` examples untouched.
- Inspect your data first: some H5/NPZ are real-only (phase flat), others (e.g. `exports/filled_arrays_noiseaware_1008Hz.npz`) have full complex P/U. Use `scripts/run_inspect_h5.py` and `scripts/run_inspect_npz.py` to see what you have before running phase/intensity/CSM steps.
- Pseudo-time isosurfaces: `scripts/run_pseudotime_isosurfaces.py` now accepts `--mesh-obj <room_scan.obj>` (plus `--mesh-stride/--mesh-maxfaces`) to overlay your scanned geometry while sweeping through pseudo-time frames.
