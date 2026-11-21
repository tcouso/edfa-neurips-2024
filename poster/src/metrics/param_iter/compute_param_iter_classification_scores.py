import yaml
import pickle
import json
import numpy as np
import pandas as pd
from typing import List
from itertools import product

from src import paths
from src.inference.utils.pandas_wrapper import PandasWrapper
from src.metrics.change_detection_metrics import classification_score, change_point_detection_score
from src.metrics.detailed_classification_scores import compute_detailed_scores

def compute_param_iter_classification_scores(selected_change_type: str) -> None:

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N: int = params[f"optim_N_{selected_change_type}"]
    k: float = params[f"optim_k_{selected_change_type}"]
    esn_features_dim: int = params["esn_features_dim"]
    param_iteration_poly_ids: List[int] = params["param_iteration_poly_ids"]
    slack_weeks_for_event_date: int = params["slack_weeks_for_event_date"]
    stable_event_types: List[str] = params["stable_event_types"]


    metadata_df = pd.read_csv(
        paths.data_processed_dir(f"metadata_annotated_filtered.csv"),
        index_col=["ID", "IDpix"],
    )
    selected_polys_metadata_df = metadata_df.loc[param_iteration_poly_ids].copy()

    if selected_change_type != "all":
        excluded_change_type = "fire" if selected_change_type == "logging" else "logging"
        selected_polys_metadata_df = selected_polys_metadata_df.loc[selected_polys_metadata_df["change_type"] != excluded_change_type]

    with open(
        paths.results_change_detections_dir(
            f"param_iteration_change_detections_{selected_band}_{esn_features_dim}_features_{selected_change_type}.pk"
        ),
        "rb",
    ) as file:
        change_detections_by_params = pickle.load(file)

    change_detection = change_detections_by_params[(N, k)]
    change_detection_df = PandasWrapper(change_detection, "change_detection")

    y_true = selected_polys_metadata_df["label"]
    y_pred = pd.Series(0, index=y_true.index)

    num_true_positives, num_actual_positives, num_model_detections = 0, 0, 0

    for pix_index in selected_polys_metadata_df.index:
        detected_changes = change_detection_df.loc[pix_index].detected_changes
        pix_metadata = metadata_df.loc[pix_index]

        y_pred.loc[pix_index] = classification_score(
                pix_metadata, 
                detected_changes, 
                slack_weeks_for_event_date
            )
        
        true_positives, actual_positives, model_detections = change_point_detection_score(
            pix_metadata, 
            detected_changes, 
            slack_weeks_for_event_date
        )

        num_true_positives += true_positives
        num_actual_positives += actual_positives
        num_model_detections += model_detections

    # Change-point detection metrics
    precision_cpd = num_true_positives / num_model_detections
    recall_cpd = num_true_positives / num_actual_positives
    f1_score_cpd = precision_cpd * recall_cpd / (0.5 * recall_cpd + 0.5 * precision_cpd)
    cpd_metrics = {
        "precision": precision_cpd,
        "recall": recall_cpd,
        "f1-score": f1_score_cpd,
    }
        
    # Compute general classification scores
    scores = compute_detailed_scores(y_true, y_pred, metadata_df=selected_polys_metadata_df)

    # Extract results from the scores dictionary
    cm = scores["confusion_matrix"]
    detailed_cm_absolutes = scores["detailed_cm_absolutes"]
    detailed_cm_percentages = scores["detailed_cm_percentages"]
    report_df = scores["classification_report"]

    # File saving

    # Save change-point detection metrics
    cpd_metrics_filename = f"param_iter_cpd_metrics_{selected_band}_{selected_change_type}.json"
    with open(paths.reports_metrics_dir(cpd_metrics_filename), "w") as file:
        json.dump(cpd_metrics, file)

    # Save detailed confusion matrix (absolute values)
    detailed_cm_absolutes_filename = f"param_iter_detailed_abs_cm_{selected_band}_{selected_change_type}.csv"
    detailed_cm_absolutes.to_csv(paths.reports_metrics_dir(detailed_cm_absolutes_filename))

    # Save detailed confusion matrix (percentages)
    detailed_cm_percentages_filename = f"param_iter_detailed_per_cm_{selected_band}_{selected_change_type}.csv"
    detailed_cm_percentages.to_csv(paths.reports_metrics_dir(detailed_cm_percentages_filename))

    # Save confusion matrix
    confusion_matrix_filename = f"param_iter_cm_{selected_band}_{selected_change_type}.npy"
    np.save(paths.reports_metrics_dir(confusion_matrix_filename), cm)

    # Save classification report
    classification_report_filename = f"param_iter_classification_report_{selected_band}_{selected_change_type}.csv"
    report_df.to_csv(paths.reports_metrics_dir(classification_report_filename), index=True)

    # Metrics by event type
    for non_change_type, change_type in product(stable_event_types, [selected_change_type]):
        stable_type_mask = (selected_polys_metadata_df["change_type"] == non_change_type)
        change_type_mask = (selected_polys_metadata_df["change_type"] == change_type)

        mask = (change_type_mask | stable_type_mask)

        y_true_subset = y_true.loc[mask]
        y_pred_subset = y_pred.loc[mask]

        metadata_subset_df = selected_polys_metadata_df.loc[mask]

        subset_scores = compute_detailed_scores(y_true_subset, y_pred_subset, metadata_df=metadata_subset_df)

        # File saving

        # Extract results from the scores dictionary
        cm_subset = subset_scores["confusion_matrix"]
        detailed_cm_absolutes_subset = subset_scores["detailed_cm_absolutes"]
        detailed_cm_percentages_subset = subset_scores["detailed_cm_percentages"]
        report_df_subset = subset_scores["classification_report"]

        # Save detailed confusion matrix (absolute values)
        detailed_cm_absolutes_filename = f"param_iter_detailed_abs_cm_{selected_band}_{non_change_type}_{change_type}.csv"
        detailed_cm_absolutes_subset.to_csv(paths.reports_metrics_dir(detailed_cm_absolutes_filename))

        # Save detailed confusion matrix (percentages)
        detailed_cm_percentages_filename = f"param_iter_detailed_per_cm_{selected_band}_{non_change_type}_{change_type}.csv"
        detailed_cm_percentages_subset.to_csv(paths.reports_metrics_dir(detailed_cm_percentages_filename))

        # Save confusion matrix
        confusion_matrix_filename = f"param_iter_cm_{selected_band}_{non_change_type}_{change_type}.npy"
        np.save(paths.reports_metrics_dir(confusion_matrix_filename), cm_subset)

        # Save classification report
        classification_report_filename = f"param_iter_classification_report_{selected_band}_{non_change_type}_{change_type}.csv"
        report_df_subset.to_csv(paths.reports_metrics_dir(classification_report_filename), index=True)
