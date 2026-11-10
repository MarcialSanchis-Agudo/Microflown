# Walk-around scanning of a furnished room with an acoustic vector probe

Code for the paper *Walk-around scanning of a furnished room with the Microflown acoustic vector probe:
noise diagnosis, Helmholtz-consistent reconstruction and energy flux*
(M. Sanchis-Agudo, Theo [surname], D. Fernández Comesaña, R. Vinuesa).

Starting from the raw Scan & Paint 3D export of one room scan, it reproduces every number, table and figure
in the paper. Each script prints the numbers it produces, and `run_all.sh` keeps that output in `results/*.log`.

## Quick start

```sh
pip install -r requirements.txt
# put the data files in data/ (see data/README.md)
sh run_all.sh          # about 1.5-2 h; the Gaussian-process fits dominate
```

Outputs: numbers in `results/*.log`; figures in `paperfigs/`; `room_wave_750Hz.gif` (animation of the
reconstructed wave); `room_flux_3d_viewer.html` (interactive 3-D view, opens in any browser).

## What is where

| Paper | Method | Code |
|---|---|---|
| Sec. 3, Stage 2a | four-feature anomaly score, top 5 % flagged | `lib/room_stage2.py: score` |
| Sec. 2.3, Stage 2b | local Helmholtz (j_l Y_lm) fit, dwell weights | `lib/room_stage2.py: local_recon` |
| Sec. 2.4 | joint pressure–velocity Gaussian process, dwell-dependent noise | `lib/room_gp.py` |
| Sec. 2.5, Stage 3 | energy flux, flux-line intersection | `analysis/run_room_pipeline.py` |
| Sec. 2.2 | exterior spherical-harmonic expansion | `lib/mf_pipeline/mf_directivity.py` |
| — | reading the HDF5 export without h5py | `lib/mini_h5.py`, `lib/room_raw.py` |
| — | analysis frame (rigid transform of the probe positions) | `lib/frame.json` |

`analysis/run_room_pipeline.py <f>` runs the processing chain at one frequency and prints its quality metrics
as one JSON line. `analysis/verify_*.py` are the validation and null tests; each docstring says what is tested
and against what. `figures/make_*.py` draw the figures from saved results.

## Using it on another scan

The processing (`score`, `local_recon`, `gp_local` / `gp_predict`) takes plain arrays: positions, complex
pressure and velocity, and averages per point. Specific to this data set:
- file name and HDF5 dataset names of the export: `lib/room_raw.py`;
- the analysis frame: `lib/frame.json` (any rigid frame works; only map orientations depend on it);
- the dwell bookkeeping (2 cm cells, `SampleCellMatrix`): `lib/room_stage2.py: dwell_blocks`.

## Conventions

The export stores transfer functions in the e^{+iωt} convention, with the velocity sign inverted relative to
the pressure; the physical flux is I = −½ Re(p u*_stored) (paper, Sec. 2.5). Pressure and velocity are transfer
functions to the first reference sensor, so levels are relative.

## Licence and data

[licence to be decided by the authors]. The measured data belong to Microflown Technologies; see `data/README.md`.
