import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

from src import paths
from src.data.utils import create_output_path


def plot_clasification_scores_over_parameters(selected_change_type: str) -> None:

    params_path=paths.config_dir("params.yaml")
    metrics_dir: Path = paths.reports_dir("metrics")
    plots_dir: Path = paths.reports_dir("figures")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]

    f1_scores_filename = "_".join(["f1_scores", selected_band, selected_change_type]) + ".npy"
    precision_scores_filename = "_".join(["precision_scores", selected_band, selected_change_type]) + ".npy"
    recall_scores_filename = "_".join(["recall_scores", selected_band, selected_change_type]) + ".npy"

    f1_scores_path = metrics_dir / f1_scores_filename
    precision_scores_path = metrics_dir / precision_scores_filename
    recall_scores_path = metrics_dir / recall_scores_filename

    f1_scores = np.load(f1_scores_path)
    precision_scores = np.load(precision_scores_path)
    recall_scores = np.load(recall_scores_path)

    metrics_dict = {
        "f1_scores": f1_scores,
        "recall_scores": recall_scores,
        "precision_scores": precision_scores,
    }

    # plt.rcParams["font.size"] = 24  # General font size
    # plt.rcParams["axes.titlesize"] = 24  # Title font size
    # plt.rcParams["axes.labelsize"] = 24  # Axis label font size
    # plt.rcParams["xtick.labelsize"] = 24  # X-tick font size
    # plt.rcParams["ytick.labelsize"] = 24  # Y-tick font size
    # plt.rcParams["figure.titlesize"] = 24  # Overall figure title font size if needed

    n = len(metrics_dict)

    plt.figure(figsize=(n * 10, 8))

    for i, (metrics_name, metrics) in enumerate(metrics_dict.items()):

        plt.subplot(1, n, i + 1)

        df = pd.DataFrame(metrics.reshape(len(N_values), len(k_values)))
        df.columns = k_values
        df.index = N_values

        sns.heatmap(
            df,
            annot=False,
            cmap="YlGnBu",
            # cbar_kws={"label": metrics_name},
            vmin=0,
            vmax=1,
        )

        plt.title(metrics_name.replace("_", " ").capitalize())
        plt.xlabel("k")
        plt.ylabel("N")

    plt.tight_layout()

    plot_filename = "_".join([selected_band, "scores_over_params", selected_change_type]) + ".png"
    plot_path = plots_dir / plot_filename

    create_output_path(plot_path)

    plt.savefig(plot_path)
    plt.close()
