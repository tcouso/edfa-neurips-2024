import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

def compute_detailed_scores(y_true, y_pred, metadata_df):

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Detailed confusion matrix
    detailed_cm = metadata_df.copy()
    detailed_cm.loc[:, "prediction"] = y_pred

    detailed_cm_absolutes = detailed_cm[["change_type", "label", "prediction"]].value_counts(normalize=False)
    detailed_cm_percentages = detailed_cm[["change_type", "label", "prediction"]].value_counts(normalize=True)

    # Classification report
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    return {
        "confusion_matrix": cm,
        "detailed_cm_absolutes": detailed_cm_absolutes,
        "detailed_cm_percentages": detailed_cm_percentages,
        "classification_report": report_df
    }

