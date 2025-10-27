# Microflown Speaker Pipeline

Tools to clean noisy sensor measurements from Microflown scans and build directivity patterns on the sphere.

All outputs are written to:
`/Users/marcialsanchis/Desktop/PhD/ModeAir/MicroFLown/mf_pipeline/exports`
(You can change paths in `mf_pipeline/mf_config.py`.)

---

## What the pipeline does

For each target frequency:

1.  Load complex pressure `P` and particle velocity `U = [Ux, Uy, Uz]` at measured sensor positions.
2.  Detect noisy sensors using local robust statistics (k-nearest neighbors, median, MAD).
3.  Recalculate *only* those noisy sensors with local complex RBF interpolation (real and imag parts separately).
4.  Choose a scalar metric from the data (for example: `|U|`, `|Ux|`, `|Uy|`, `|Uz|`, or `|P|`).
5.  Fit a spherical-harmonics (SH) model of that metric on the sphere (with Tikhonov regularization).
6.  Evaluate the fitted model on a dense sphere to export:
    * equirectangular map (lon/lat image),
    * horizontal and vertical polar cuts.
7.  Produce diagnostics to show what changed per sensor (cloud plots, per-point deltas, histogram, raw vs filled scatter).

> **Note:** We clean sensors, not a speaker mesh.

---

## Repository layout

mf_pipeline/ ├── mf_config.py # absolute paths (H5 and EXPORTS) ├── mf_fill.py # noisy-sensor detection + interpolation, saves NPZ ├── mf_directivity.py # SH fit + evaluation, polar helpers ├── mf_plots.py # cloud plots and per-point delta plots ├── run_fill.py # CLI step 1 ├── run_cloud_plots.py # CLI step 2 └── run_directivity.py # CLI step 3
## Install

---

## Install

1.  Use your existing conda env (example: `surd`) and install Python deps:

    ```bash
    conda activate surd
    pip install numpy scipy h5py matplotlib
    ```

2.  Configure absolute paths in `mf_pipeline/mf_config.py`:

    ```python
    from pathlib import Path

    EXPORTS = Path("/Users/marcialsanchis/Desktop/PhD/ModeAir/MicroFLown/mf_pipeline/exports")
    EXPORTS.mkdir(parents=True, exist_ok=True)

    H5_PATH  = Path("/Users/marcialsanchis/Desktop/PhD/ModeAir/MicroFLown/exports") / "Loudspeaker_20mm.h5"
    H5_GROUP = "/Proc-2_3D_Cuboid,20_mm_resolution"
    ```

---

## Quick start

1.  **Clean one frequency** (interpolate only flagged sensors):

    ```bash
    python mf_pipeline/run_fill.py --f0 1007.8
    ```

    This creates the NPZ:
    `.../mf_pipeline/exports/filled_arrays_noiseaware_1008Hz.npz`

2.  **Point-cloud and per-point delta plots:**

    ```bash
    python mf_pipeline/run_cloud_plots.py \
      --npz /Users/marcialsanchis/Desktop/PhD/ModeAir/MicroFLown/mf_pipeline/exports/filled_arrays_noiseaware_1008Hz.npz \
      --metric u_mag
    ```
    (Available metrics: `u_mag` | `ux` | `uy` | `uz` | `p`)

3.  **Directivity map, sphere arrays, and polar cuts:**

    ```bash
    python mf_pipeline/run_directivity.py \
      --npz /Users/marcialsanchis/Desktop/PhD/ModeAir/MicroFLown/mf_pipeline/exports/filled_arrays_noiseaware_1008Hz.npz \
      --metric u_mag --db
    ```
    (Add `--db` to normalize and plot in dB.)

---

## How it works (plain language)

### Noisy sensor detection

1.  For each sensor, compare its value to its local neighborhood (k nearest positions).
2.  Compute local median and median absolute deviation (MAD).
3.  If the sensor deviates too much from its neighbors (robust z-score > `tau`), mark it noisy.

### Recalculation (only for flagged sensors)

1.  Interpolate complex values using nearby *clean* sensors.
2.  Do real and imaginary parts separately with a smooth RBF (multiquadric).
3.  Non-flagged sensors are kept as-is.

### Per-point delta

For the chosen metric (for example `|U|`), compute $\text{delta}_i = \text{value\_filled}_i - \text{value\_raw}_i$. We store plots/CSV so you can see which sensors changed and by how much.

### Spherical harmonics (SH)

Fit a real SH expansion up to degree `lmax` with Tikhonov regularization (`lambda`). Evaluate the fitted model on a longitude/latitude grid (map) and also on a true theta/phi grid (sphere). Export horizontal and vertical polar cuts from the fitted model.

---

## CLI reference

### `run_fill.py`

**Purpose:** detect noisy sensors and interpolate just those; save a filled NPZ.

* **Arguments:**
    * `--f0` (float, required): target frequency in Hz.
    * `--tau` (float, default 3.5): robust z-score threshold.
    * `--k-nn` (int, default 12): number of neighbors for local statistics.
    * `--smooth` (float, default 0.2): RBF smoothing.
    * `--tag` (str, default "noiseaware"): output tag.
* **Output:**
    * `filled_arrays_{tag}_{Hz}.npz` in `EXPORTS`.
* **Keys in NPZ:** `p`, `u` (filled complex), `p_raw`, `u_raw` (raw complex), `flags` dict (p, ux, uy, uz boolean masks), `pos` (positions), `meta` (f0, tau, k_nn, smooth, tag).

### `run_cloud_plots.py`

**Purpose:** visualize changes at each sensor.

* **Arguments:**
    * `--npz`: path to the filled NPZ.
    * `--metric`: one of `u_mag` | `ux` | `uy` | `uz` | `p`.
* **Outputs in EXPORTS:**
    * `*_cloud_before_after.png` (3D: raw, filled, difference).
    * `*_residual_hist.png` (histogram of deltas).
    * `*_delta_series.png` (index-wise delta and sorted absolute delta).
    * `*_raw_vs_filled.png` (scatter with y=x).
    * `*_delta.csv` (index, raw, filled, delta, abs_delta, is_noisy).

### `run_directivity.py`

**Purpose:** build directivity map and polar cuts from the filled NPZ.

* **Arguments:**
    * `--npz`: path to the filled NPZ.
    * `--metric`: one of `u_mag` | `ux` | `uy` | `uz` | `p`.
    * `--lmax` (int, default 8): SH degree.
    * `--lam` (float, default 1e-3): Tikhonov regularization.
* `--res_lon` (int, default 361): longitudinal samples for maps.
* `--res_lat` (int, default 181): latitudinal samples for maps.
    * `--db` (flag): normalize and plot in dB.
* **Outputs in EXPORTS:**
    * `*_directivity.png` (equirectangular map).
    * `*_map_lonlat.npz` (lon\_deg, lat\_deg, V).
    * `*_sphere_theta_phi.npz` (theta, phi, V; radians).
    * `*_polar_horizontal_0deg.png`, `*_polar_vertical_az0deg.png`.

---

## Typical outputs (example for ~1008 Hz, metric `u_mag`)

* `filled_arrays_noiseaware_1008Hz.npz`
* `filled_arrays_noiseaware_1008Hz_u_mag_cloud_before_after.png`
* `filled_arrays_noiseaware_1008Hz_u_mag_residual_hist.png`
* `filled_arrays_noiseaware_1008Hz_u_mag_delta_series.png`
* `filled_arrays_noiseaware_1008Hz_u_mag_raw_vs_filled.png`
* `filled_arrays_noiseaware_1008Hz_u_mag_delta.csv`
* `filled_arrays_noiseaware_1008Hz_u_mag_directivity.png`
* `filled_arrays_noiseaware_1008Hz_u_mag_map_lonlat.npz`
* `filled_arrays_noiseaware_1008Hz_u_mag_sphere_theta_phi.npz`
* `filled_arrays_noiseaware_1008Hz_u_mag_polar_horizontal_0deg.png`
* `filled_arrays_noiseaware_1008Hz_u_mag_polar_vertical_az0deg.png`

---

## Parameters and tips

* **`k-nn` (neighbors):** 12 to 24. Too small = unstable med/MAD; too large = detection becomes blunt.
* **`tau` (outlier threshold):** 3.0 to 4.0. Increase to reduce false positives.
* **`smooth` (RBF smoothing):** 0.15 to 0.3. Increase if filled values look wiggly.
* **`lmax` (SH degree):** sparse scans 4 to 6; denser scans 8 to 12.
* **`lam` (regularization):** 1e-4 to 1e-2. Raise if you see ringing or overfit.
* **`metric`:** `u_mag` (default) emphasizes particle velocity; `p` behaves more like SPL.

**Good practice:**

* Interpolate real and imaginary parts separately. Do not interpolate magnitudes directly.
* If you know the acoustic center, use it for better angular symmetry.
* Match `lmax` to spatial coverage; use `lam` to stabilize.

---

## Troubleshooting

* **File not found:** verify absolute paths in `mf_pipeline/mf_config.py` (`H5_PATH` and `EXPORTS`).
* **Import errors:** run from the project root; the `run_*` scripts include small `sys.path` shims.
* **Delta plots empty:** re-run `run_fill.py` so the NPZ includes `p_raw`, `u_raw`, and `flags`.
* **Noisy mask not highlighted:** confirm `flags` exists in the NPZ.

---

## License

Add your preferred license here.

---

## Acknowledgments

This pipeline is inspired by scanning-based directivity measurement workflows and is designed to highlight sensor-centric cleaning before spherical extrapolation.
