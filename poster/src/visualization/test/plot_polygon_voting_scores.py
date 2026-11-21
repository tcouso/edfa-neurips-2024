import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay
from src import paths


def plot_polygon_voting_scores() -> None:
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    voting_threshold = params["voting_threshold"]

    sweep_df = pd.read_csv(
        paths.reports_metrics_dir(f"polygon_voting_threshold_sweep_{selected_band}.csv"), index_col="threshold"
    )
    by_event_type_df = pd.read_csv(
        paths.reports_metrics_dir(f"polygon_voting_by_event_type_{selected_band}.csv")
    )

    fig, ax = plt.subplots()
    sweep_df.plot(ax=ax, marker="o")
    ax.axvline(x=voting_threshold, color="gray", linestyle="--")
    ax.set_xlabel("Voting threshold")
    ax.set_ylabel("Score")
    fig.savefig(paths.reports_figures_dir(f"polygon_voting_threshold_sweep_{selected_band}.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (_, row) in zip(axes.ravel(), by_event_type_df.iterrows()):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
        ax.set_title(f"{row['change_type']} vs {row['non_change_type']} (accuracy {row['accuracy']:.2f})")
    fig.suptitle(f"Polygon voting, threshold = {voting_threshold}")
    fig.savefig(paths.reports_figures_dir(f"polygon_voting_confusion_matrices_{selected_band}.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_polygon_voting_scores()
