# Walk-around scanning of a furnished room with an acoustic vector probe

Code for the paper *Walk-around scanning of a furnished room with the Microflown acoustic vector probe:
noise diagnosis, Helmholtz-consistent reconstruction and energy flux*
(M. Sanchis-Agudo, Theo [surname], D. Fernández Comesaña, R. Vinuesa).

Starting from the raw Scan & Paint 3D export of one room scan, it reproduces every number, table and figure
in the paper. Each script prints the numbers it produces, and `run_all.sh` keeps that output in `results/*.log`.

## Requirements

Tested with Python 3.10, NumPy 2.2.6, SciPy 1.15.2, Matplotlib 3.10.6 and Pillow 11.3.0 (conda).
SciPy must be older than 1.17 (`scipy.special.sph_harm` is used). No other packages are needed: the HDF5
export is read by `lib/mini_h5.py`, without h5py.

## Running it

1. Create an environment and install the packages (conda shown; a venv works the same way):

   ```sh
   conda create -n roomscan python=3.10
   conda activate roomscan
   python -m pip install -r requirements.txt
   ```

2. Put the data files in `data/` (see `data/README.md`). Only `Martial measurements.psd` is needed for the
   numbers; the room model and photographs are needed for Fig. 1 and the registration check.

3. From this folder (`room_scan_code/`), run

   ```sh
   PYTHON=$(which python) sh run_all.sh
   ```

   `PYTHON` selects the interpreter of the active environment (default `python3`, which on many systems is not
   the conda one). The script first checks that the packages import, then runs every step, writes each step's
   output to `results/<step>.log`, and stops at the first error, naming the log to read.

Run time: about 1.5–2 h on a laptop; the Gaussian-process fits dominate.

Outputs: numbers in `results/*.log`; figures in `paperfigs/`; `room_wave_750Hz.gif` (animation of the
reconstructed wave); `room_flux_3d_viewer.html` (interactive 3-D view, opens in any browser).

Results are deterministic: nearest-neighbour ties (frequent on the 2 cm grid) are broken by point index, and
every random draw has a fixed seed.

## What is where

| Paper | Method | Code |
|---|---|---|
| Sec. 3, Eq. 12 | four-feature anomaly score, top 5 % flagged | `lib/room_stage2.py: score` |
| Sec. 2.3, Eq. 4 | local Helmholtz (j_l Y_lm) fit, dwell weights | `lib/room_stage2.py: local_recon` |
| Sec. 2.4, Eqs. 5–9 | joint pressure–velocity Gaussian process, dwell-dependent noise | `lib/room_gp.py` |
| Sec. 2.5, Eq. 10 | energy flux, flux-line intersection | `analysis/run_room_pipeline.py` |
| Sec. 2.2, Eqs. 2–3 | exterior spherical-harmonic expansion | `lib/mf_pipeline/mf_directivity.py` |
| Sec. 2.6, Eq. 11 | random error vs averages, reference-pair test | `analysis/verify_room_refpair.py`, `verify_room_noise_model.py` |
| — | reading the HDF5 export without h5py | `lib/mini_h5.py`, `lib/room_raw.py` |
| — | analysis frame (rigid transform of the probe positions) | `lib/frame.json` |

`analysis/run_room_pipeline.py <f>` runs the processing chain at one frequency and prints its quality metrics
as one JSON line. `analysis/verify_*.py` are the validation and null tests; each docstring says what is tested
and against what (`verify_room_gp_kernel.py` and `verify_room_gp_closedform.py` check the covariances of
Eq. 6 against finite differences). `figures/make_*.py` draw the figures from saved results.

## Using it on another scan

The processing (`score`, `local_recon`, `gp_local` / `gp_predict`) takes plain arrays: positions, complex
pressure and velocity, and averages per point. Specific to this data set:
- file name and HDF5 dataset names of the export: `lib/room_raw.py`;
- the analysis frame: `lib/frame.json` (any rigid frame works; only map orientations depend on it);
- the dwell bookkeeping (2 cm cells, `SampleCellMatrix`): `lib/room_stage2.py: dwell_blocks`.

## Conventions

The export stores transfer functions in the e^{+iωt} convention, with the velocity sign inverted relative to
the pressure; the physical flux is I = −½ Re(p u*_stored) (paper, Sec. 2.5). Pressure and velocity are transfer
functions to the first reference sensor, so levels are relative. Averages per cell are counted as recorded
samples / 2048 (no overlap assumed; the export does not record the averaging scheme).

## Licence and data

[licence to be decided by the authors]. The measured data belong to Microflown Technologies and are available
from the authors on request, subject to agreement with Microflown; see `data/README.md`.
