import itertools
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from src import paths


def voting_groupby(y_pred, th):
    return y_pred["0"].groupby("ID").apply(lambda x: (x.sum() / len(x)) > th).astype(int)


def compute_polygon_voting_scores() -> None:
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    param_iteration_poly_ids = params["param_iteration_poly_ids"]
    voting_threshold = params["voting_threshold"]

    metadata_df = pd.read_csv(
        paths.data_processed_dir("metadata_annotated_filtered.csv"), index_col=["ID", "IDpix"]
    )
    y_pred = pd.read_csv(paths.results_dir(f"y_pred_{selected_band}.csv"), index_col=["ID", "IDpix"])
    y_true = pd.read_csv(paths.results_dir(f"y_true_{selected_band}.csv"), index_col=["ID", "IDpix"])

    poly_y_true = y_true["label"].groupby("ID").first()

    sweep_rows = []
    for th in np.arange(0, 0.55, 0.05):
        poly_y_pred = voting_groupby(y_pred, th)
        sweep_rows.append({
            "threshold": round(th, 2),
            "accuracy": accuracy_score(poly_y_true, poly_y_pred),
            "precision": precision_score(poly_y_true, poly_y_pred, zero_division=0),
            "recall": recall_score(poly_y_true, poly_y_pred, zero_division=0),
            "f1-score": f1_score(poly_y_true, poly_y_pred, zero_division=0),
        })
    pd.DataFrame(sweep_rows).to_csv(
        paths.reports_metrics_dir(f"polygon_voting_threshold_sweep_{selected_band}.csv"), index=False
    )

    poly_y_pred = voting_groupby(y_pred, voting_threshold)
    report = classification_report(poly_y_true, poly_y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        paths.reports_metrics_dir(f"polygon_voting_classification_report_{selected_band}.csv")
    )

    poly_metadata_df = metadata_df["change_type"].groupby("ID").first()
    poly_metadata_df = poly_metadata_df.drop(param_iteration_poly_ids)
    poly_detections_df = pd.concat([poly_metadata_df, poly_y_true, poly_y_pred], axis=1)
    poly_detections_df.columns = ["change_type", "true", "pred"]

    by_event_type_rows = []
    for change_type, non_change_type in itertools.product(["fire", "logging"], ["stable", "drought"]):
        mask = poly_detections_df["change_type"].isin([change_type, non_change_type])
        sel_y_true = poly_detections_df["true"][mask]
        sel_y_pred = poly_detections_df["pred"][mask]
        tn, fp, fn, tp = confusion_matrix(sel_y_true, sel_y_pred, labels=[0, 1]).ravel()
        by_event_type_rows.append({
            "non_change_type": non_change_type,
            "change_type": change_type,
            "accuracy": accuracy_score(sel_y_true, sel_y_pred),
            "f1-score": f1_score(sel_y_true, sel_y_pred, zero_division=0),
            "recall": recall_score(sel_y_true, sel_y_pred, zero_division=0),
            "precision": precision_score(sel_y_true, sel_y_pred, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })
    pd.DataFrame(by_event_type_rows).to_csv(
        paths.reports_metrics_dir(f"polygon_voting_by_event_type_{selected_band}.csv"), index=False
    )


if __name__ == "__main__":
    compute_polygon_voting_scores()
