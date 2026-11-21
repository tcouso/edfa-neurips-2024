# esn-change-detection

Reference implementation of the method in [*Enhanced Detection of Human-Driven Forest Alterations using Echo State Networks*](https://neurips.cc/virtual/2024/100568) (Couso Coddou, Aguirre, Carrasco, Lopatin), NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning.

This repository is an extract of the author's original development repository, taken at the state that corresponds to the experiments reported in the paper. Commit history, exploratory notebooks, trained models and results were left out, and the dataset is not included because it is not public.

## Method as implemented

- `src/features/denoise_ndvi_signal.py`: resampling, moving standard-deviation filter and Holt-Winters smoothing.
- `src/data/esn_fault_detection_partition.py`: random polygon-level split (`esn_training_percentage`) between ESN training and change detection.
- `src/models/train_esn.py`: non-feedback ESN with an RLS readout (ReservoirPy).
- `src/models/fault_detection.py`: per-pixel lower bound, step-by-step retraining and the N-consecutive-below-`k * lower_bound` rule.
- `src/models/predict_over_polygons.py`: polygon voting with threshold `th`.

Hyperparameters are in `config/params.yaml`.

## Differences with the paper text

- The denoising step resamples to weekly intervals (`resample("W")`), not bi-weekly. The 104-feature window is two years of weekly data.
- Only the non-feedback ESN is implemented.

## Data

The dataset is not distributed. `src/data/join_signal_to_aux_datasets.py` documents the expected input under `data/raw/`: one directory per event type (`estable-arboreo-ts`, `sequia-arboreo-ts`, `tala-arboreo-ts`, `incendio-arboreo-ts`), each with `<PREFIX>_TimeSerie_<band>.csv` (one row per pixel, one column per date) and `<PREFIX>_AuxiliarFix.csv` (pixel and polygon metadata). Column names are mapped in `config/params.yaml`.

## Run

```bash
pip install -e .
pip install -r requirements.txt
bash run.sh
```

`TELEGRAM_TOKEN` and `CHAT_ID` may be set in a `.env` file for progress notifications. They are optional.
