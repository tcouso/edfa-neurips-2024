import pandas as pd
from typing import Tuple

def classification_score(
    pix_metadata: pd.DataFrame,
    detected_changes: pd.Series,
    slack_weeks_for_event_date: int,
) -> int:

    if pix_metadata["label"] == 1:

        lb_detection_date = pd.Timestamp(pix_metadata["start"]) - pd.Timedelta(weeks=slack_weeks_for_event_date)
        ub_detection_date = pd.Timestamp(pix_metadata["end"]) + pd.Timedelta(weeks=slack_weeks_for_event_date)

        change_detected_in_date_range = (
            detected_changes[
                (detected_changes.index < ub_detection_date)
                & (detected_changes.index > lb_detection_date)
            ]
            == 1
        ).any()

        pix_detection = 1 if change_detected_in_date_range else 0
        

    elif pix_metadata["label"] == 0:

        change_detected = (detected_changes == 1).any()

        pix_detection = 1 if change_detected else 0

    return pix_detection


def change_point_detection_score(
    pix_metadata: pd.DataFrame,
    detected_changes: pd.Series,
    slack_weeks_for_event_date: int,
) -> Tuple[int, int, int]:
    
    true_positives = 1
    actual_positives = 1
    model_detections = 1

    if pix_metadata["label"] == 1:

        actual_positives += 1 # One change per pixel
        lb_detection_date = pd.Timestamp(pix_metadata["start"]) - pd.Timedelta(weeks=slack_weeks_for_event_date)
        ub_detection_date = pd.Timestamp(pix_metadata["end"]) + pd.Timedelta(weeks=slack_weeks_for_event_date)

        change_detected_in_date_range = (
            detected_changes[
                (detected_changes.index < ub_detection_date)
                & (detected_changes.index > lb_detection_date)
            ]
            == 1
        ).any()


        num_of_changes_detected_out_date_range = (
        detected_changes[
            (detected_changes.index > ub_detection_date)
            | (detected_changes.index < lb_detection_date)
        ]
        == 1
        ).sum()

        num_of_changes_detected_in_date_range = 1 if change_detected_in_date_range else 0
        true_positives += num_of_changes_detected_in_date_range
        model_detections += num_of_changes_detected_in_date_range + num_of_changes_detected_out_date_range

    elif pix_metadata["label"] == 0:

        num_of_detected_changes = (detected_changes == 1).sum()
        model_detections += num_of_detected_changes

    return true_positives, actual_positives, model_detections
