#!/bin/bash

cd "$(dirname "$0")"

python3 src/data/make_dataset.py
python3 src/features/build_features.py

python3 src/training/train_model.py
python3 src/inference/simulate_pixels.py

python3 src/inference/iterate_parameters.py
python3 src/metrics/param_iter/compute_param_iter_metrics.py
python3 src/visualization/param_iter/build_param_iter_visualizations.py

python3 src/inference/predict_model.py
python3 src/metrics/test/compute_test_metrics.py
python3 src/visualization/test/build_test_visualizations.py

python3 src/metrics/test/compute_polygon_voting_scores.py
python3 src/visualization/test/plot_polygon_voting_scores.py
python3 src/visualization/test/plot_pixel_detections.py
