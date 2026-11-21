import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
)

from src import paths


def plot_detailed_param_iter_scores(selected_change_type: str):
    params_path: Path = paths.config_dir("params.yaml")
    metrics_dir: Path = paths.reports_dir("metrics")
    figures_dir: Path = paths.reports_dir("figures")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]

    confusion_matrix_filename = f"param_iter_cm_{selected_band}_{selected_change_type}.npy"
    confusion_matrix_path = metrics_dir / confusion_matrix_filename

    cm = np.load(confusion_matrix_path)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot()

    cm_display_filename = "_".join(
        [
            "param_iter",
            "cm",
            selected_band,
            selected_change_type,
        ]
    )

    cm_display_filename += ".png"

    cm_display_path = figures_dir / cm_display_filename
    cm_display.figure_.savefig(cm_display_path)
