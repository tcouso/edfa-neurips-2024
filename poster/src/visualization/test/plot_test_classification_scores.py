import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
)
from itertools import product
from typing import List

from src import paths


def plot_test_classification_scores():
    params_path: Path = paths.config_dir("params.yaml")
    metrics_dir: Path = paths.reports_dir("metrics")
    figures_dir: Path = paths.reports_dir("figures")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]

    confusion_matrix_filename = "_".join(
        [
            "test",
            "cm",
            selected_band,
        ]
    )
    confusion_matrix_filename += ".npy"
    confusion_matrix_path = metrics_dir / confusion_matrix_filename

    cm = np.load(confusion_matrix_path)
    cm_display = ConfusionMatrixDisplay(cm)

    cm_display.plot()

    cm_display_filename = "_".join(
        [
            "test",
            "cm",
            selected_band,
        ]
    )

    cm_display_filename += ".png"

    cm_display_path = figures_dir / cm_display_filename
    cm_display.figure_.savefig(cm_display_path)


def plot_test_classification_scores_by_event_type():

    params_path: Path = paths.config_dir("params.yaml")
    metrics_dir: Path = paths.reports_dir("metrics")
    figures_dir: Path = paths.reports_dir("figures")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    stable_event_types: List[str] = params["stable_event_types"]
    change_event_types: List[str] = params["change_event_types"]

    for non_change_type, change_type in product(stable_event_types, change_event_types):

        confusion_matrix_filename = "_".join(
            [
                "test",
                "cm",
                selected_band,
                non_change_type,
                change_type,
            ]
        )
        confusion_matrix_filename += ".npy"
        confusion_matrix_path = metrics_dir / confusion_matrix_filename

        cm = np.load(confusion_matrix_path)

        cm_display = ConfusionMatrixDisplay(cm)

        cm_display.plot()

        cm_display_filename = "_".join(
            [
                "test",
                "cm",
                selected_band,
                non_change_type,
                change_type,
            ]
        )
        cm_display_filename += ".png"

        cm_display_path = figures_dir / cm_display_filename
        cm_display.figure_.savefig(cm_display_path)
