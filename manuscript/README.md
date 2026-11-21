# Manuscript version

Code for the method as described in the [workshop manuscript](https://ccai-papers.s3.us-east-1.amazonaws.com/neurips2024/77/paper.pdf) of [*Enhanced Detection of Human-Driven Forest Alterations using Echo State Networks*](https://neurips.cc/virtual/2024/100568). This is the earlier version of the method: the most recent one, presented in the workshop poster, is in [`../poster`](../poster).

This directory is an extract of the author's original development repository, taken at the state closest to the experiments reported in the manuscript. Commit history, exploratory notebooks, trained models and results were left out, and the dataset is not included because it is not public.

## Method

For each pixel:

1. The NDVI series is denoised (resampling, moving standard-deviation filter, Holt-Winters smoothing).
2. An Echo State Network, trained beforehand on pixels from stable and drought polygons, is fine-tuned on the pixel's series up to one year before its end. Its forecast of the remaining year is the pixel's reference lower bound.
3. The ESN is then updated one step at a time over that last year, producing a time-shifted forecast at each step.
4. The pixel is flagged as changed if N consecutive forecasted values fall below `k` times the lower bound.

A polygon is labelled as changed if the fraction of flagged pixels is at least a voting threshold `th`. The parameters `N`, `k` and `th` are selected by grid search over F1-score (`config/params.yaml`).

The ESN has 500 reservoir units, leaking rate 0.5, spectral radius 0.9 and an RLS readout (ReservoirPy), and predicts the next NDVI value from the previous 104 weekly values (two years).

## Repository structure

```
config/params.yaml                          all hyperparameters and dataset conventions
src/data/make_synthetic_dataset.py          synthetic dataset in the expected raw format
src/data/join_signal_to_aux_datasets.py     merge raw signal and polygon metadata
src/data/esn_fault_detection_partition.py   random polygon split: ESN training vs. change detection
src/features/denoise_ndvi_signal.py         denoising pipeline
src/features/build_features.py              denoise the ESN training signals
src/features/slice_time_series.py           feature/target windows
src/models/train_esn.py                     ESN training and regression metrics
src/models/fault_detection.py               lower bound, time-shifted forecasts, detection rule
src/models/run_simulations.py               per-pixel simulation
src/models/predict_over_pixels.py           pixel predictions over the (N, k) grid
src/models/predict_over_polygons.py         polygon voting over the (N, k, th) grid
src/metrics/                                classification metrics over the parameter grid
src/visualization/                          metric plots
run.sh                                      full pipeline, from data preparation to plots
```

## Data

The dataset is not distributed. The pipeline expects one directory per event type under `data/raw/` (`stable`, `drought`, `logging`, `fire`), each with:

- `signal_<band>.csv`: one row per pixel, with an `IDpix` column and one column per acquisition date (`YYYY-MM-DD`) holding the band value. Missing observations are empty.
- `metadata.csv`: one row per pixel, with columns `ID` (polygon), `IDpix`, `lat`, `lon`, `change_start` and `last_non_change_date` (`DD-MM-YYYY`, empty for stable and drought polygons) and `vegetation_type` (`native`, `exotic` or `mixed`).

To check the pipeline without the real data, generate a synthetic dataset in this format:

```bash
python src/data/make_synthetic_dataset.py
```

It writes 20 polygons of 5 pixels each (seasonal NDVI with noise, gaps and outliers; abrupt drop for logging, drop and recovery for fire, a mild dip for drought). It is only meant to exercise the code: results on it are not meaningful. Setting `test_at_scale: True` in `config/params.yaml` samples `test_at_scale_sample_size` pixels per event type for a quick run.

## Run

```bash
pip install -e .
pip install -r requirements.txt
bash run.sh
```

`run.sh` runs the data preparation, ESN training, change detection, metrics (`reports/metrics/`) and plots (`reports/figures/`). Intermediate files go to `data/interim/` and `data/processed/`, and the trained ESN to `models/`; these directories are kept in the repository empty and their contents are git-ignored.

## Notes on the paper

- The denoising step resamples to weekly intervals (`resample("W")`); the paper text says bi-weekly.
- Only the non-feedback ESN is implemented.
