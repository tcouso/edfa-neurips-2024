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

def compute_test_classification_scores():

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    esn_features_dim = params["esn_features_dim"]
    poly_ids_to_exclude = params["param_iteration_poly_ids"]
    N: int = params["optim_N"]
    k: int = params["optim_k"]
    param_iteration_poly_ids: List[int] = params["param_iteration_poly_ids"]
    slack_weeks_for_event_date: int = params["slack_weeks_for_event_date"]

    metadata_df = pd.read_csv(
        paths.data_processed_dir(f"metadata_annotated_filtered.csv"),
        index_col=["ID", "IDpix"],
    )
    poly_ids = metadata_df.index.get_level_values(0).unique()
    results_dict = {}

    for poly_id in poly_ids:
        if poly_id in poly_ids_to_exclude:
            continue
        

        results_filename = f"change_detections_{selected_band}_poly_{poly_id}_pixels_{esn_features_dim}_features.pk"
        results_path = paths.results_optimal_change_detections_dir(results_filename)

        with open(results_path, "rb") as file:
            poly_change_detections_by_params = pickle.load(file)

        poly_change_detections = poly_change_detections_by_params[(N, k)]
        results_dict.update(poly_change_detections)

    change_detection_df = PandasWrapper(results_dict, "change_detection")
    selected_polys_metadata_df = metadata_df.loc[~metadata_df.index.get_level_values(0).isin(param_iteration_poly_ids)].copy()

    y_true = selected_polys_metadata_df["label"]
    y_pred = pd.Series(0, index=y_true.index)

    num_true_positives, num_actual_positives, num_model_detections = 0, 0, 0

    for pix_index in y_pred.index:
        detected_changes = change_detection_df.loc[pix_index].detected_changes
        pix_metadata = metadata_df.loc[pix_index]

        y_pred.loc[pix_index] = classification_score(
            pix_metadata, 
            detected_changes,
            slack_weeks_for_event_date,
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

    y_true = y_true.sort_index()
    y_pred = y_pred.sort_index()

    # Save predictions
    y_true.to_csv(paths.results_dir(f"y_true_{selected_band}.csv"))
    y_pred.to_csv(paths.results_dir(f"y_pred_{selected_band}.csv"))

    scores = compute_detailed_scores(y_true, y_pred, metadata_df=selected_polys_metadata_df)

    # Extract results from the scores dictionary
    cm = scores["confusion_matrix"]
    detailed_cm_absolutes = scores["detailed_cm_absolutes"]
    detailed_cm_percentages = scores["detailed_cm_percentages"]
    report_df = scores["classification_report"]

    # File saving

    # Save change-point detection metrics
    cpd_metrics_filename = f"test_cpd_metrics_{selected_band}.json"
    with open(paths.reports_metrics_dir(cpd_metrics_filename), "w") as file:
        json.dump(cpd_metrics, file)


    # Save detailed confusion matrix (absolute values)
    detailed_cm_absolutes_filename = f"test_detailed_abs_cm_{selected_band}.csv"
    detailed_cm_absolutes.to_csv(paths.reports_metrics_dir(detailed_cm_absolutes_filename))

    # Save detailed confusion matrix (percentages)
    detailed_cm_percentages_filename = f"test_detailed_per_cm_{selected_band}.csv"
    detailed_cm_percentages.to_csv(paths.reports_metrics_dir(detailed_cm_percentages_filename))

    # Save confusion matrix
    confusion_matrix_filename = f"test_cm_{selected_band}.npy"
    np.save(paths.reports_metrics_dir(confusion_matrix_filename), cm)

    # Save classification report
    classification_report_filename = f"test_classification_report_{selected_band}.csv"
    report_df.to_csv(paths.reports_metrics_dir(classification_report_filename), index=True)

    # Metrics by event type
    stable_event_types: List[str] = params["stable_event_types"]
    change_event_types: List[str] = params["change_event_types"]

    for non_change_type, change_type in product(stable_event_types, change_event_types):
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
        detailed_cm_absolutes_filename = f"test_detailed_abs_cm_{selected_band}_{non_change_type}_{change_type}.csv"
        detailed_cm_absolutes_subset.to_csv(paths.reports_metrics_dir(detailed_cm_absolutes_filename))

        # Save detailed confusion matrix (percentages)
        detailed_cm_percentages_filename = f"test_detailed_per_cm_{selected_band}_{non_change_type}_{change_type}.csv"
        detailed_cm_percentages_subset.to_csv(paths.reports_metrics_dir(detailed_cm_percentages_filename))

        # Save confusion matrix
        confusion_matrix_filename = f"test_cm_{selected_band}_{non_change_type}_{change_type}.npy"
        np.save(paths.reports_metrics_dir(confusion_matrix_filename), cm_subset)

        # Save classification report
        classification_report_filename = f"test_classification_report_{selected_band}_{non_change_type}_{change_type}.csv"
        report_df_subset.to_csv(paths.reports_metrics_dir(classification_report_filename), index=True)
