
import yaml
import pickle
import numpy as np
import pandas as pd
from typing import List
from itertools import product
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from src import paths
from src.inference.utils.pandas_wrapper import PandasWrapper
from src.metrics.change_detection_metrics import change_point_detection_score, classification_score


def enumerated_product(*args):

    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def compute_param_iter_scores(selected_change_type: str) -> None:

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]
    esn_features_dim: int = params["esn_features_dim"]
    param_iteration_poly_ids: List[int] = params["param_iteration_poly_ids"]
    slack_weeks_for_event_date: int = params["slack_weeks_for_event_date"]

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

    num_N_values = len(N_values)
    num_k_values = len(k_values)

    # Change point detection (CPD) metrics
    recall_scores_cpd = np.zeros((num_N_values, num_k_values))
    precision_scores_cpd = np.zeros((num_N_values, num_k_values))
    f1_scores_cpd = np.zeros((num_N_values, num_k_values))

    # Classification metrics
    acc_scores = np.zeros((num_N_values, num_k_values))
    recall_scores = np.zeros((num_N_values, num_k_values))
    precision_scores = np.zeros((num_N_values, num_k_values))
    f1_scores = np.zeros((num_N_values, num_k_values))


    for indices, values in enumerated_product(N_values, k_values):
        N, k = values
        N_index, k_index = indices

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

        precision_cpd = num_true_positives / num_model_detections
        recall_cpd = num_true_positives / num_actual_positives
        f1_score_cpd = precision_cpd * recall_cpd / (0.5 * recall_cpd + 0.5 * precision_cpd)
        
        recall_scores_cpd[N_index][k_index] = recall_cpd
        precision_scores_cpd[N_index][k_index] = precision_cpd
        f1_scores_cpd[N_index][k_index] = f1_score_cpd

        acc_scores[N_index][k_index] = accuracy_score(y_true, y_pred)
        recall_scores[N_index][k_index] = recall_score(y_true, y_pred)
        f1_scores[N_index][k_index] = f1_score(y_true, y_pred)
        precision_scores[N_index][k_index] = precision_score(y_true, y_pred, zero_division=0)

    f1_scores_filename = "_".join(["f1_scores", selected_band, selected_change_type])
    f1_scores_filename += ".npy"

    precision_scores_filename = "_".join(["precision_scores", selected_band, selected_change_type])
    precision_scores_filename += ".npy"

    recall_scores_filename = "_".join(["recall_scores", selected_band, selected_change_type])
    recall_scores_filename += ".npy"

    # np.save(paths.reports_metrics_dir(f1_scores_filename), f1_scores_cpd)
    # np.save(paths.reports_metrics_dir(precision_scores_filename), precision_scores_cpd)
    # np.save(paths.reports_metrics_dir(recall_scores_filename), recall_scores_cpd)

    np.save(paths.reports_metrics_dir(f1_scores_filename), f1_scores)
    np.save(paths.reports_metrics_dir(precision_scores_filename), precision_scores)
    np.save(paths.reports_metrics_dir(recall_scores_filename), recall_scores)
