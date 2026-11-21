import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import root_mean_squared_error
from src import paths
from src.inference.simulation.simulation_data_structures import get_db_instance
from src.inference.change_detection.change_detection import detect_changes_in_pixel

PIXELS_PER_POLYGON = 2


def plot_pixel_detections() -> None:
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    param_iteration_poly_ids = params["param_iteration_poly_ids"]
    slack_weeks_for_event_date = params["slack_weeks_for_event_date"]
    N = params["optim_N"]
    k = params["optim_k"]
    majority_threshold = params["majority_threshold"]

    metadata_df = pd.read_csv(
        paths.data_processed_dir("metadata_annotated_filtered.csv"), index_col=["ID", "IDpix"]
    )
    db = get_db_instance()

    for poly_id in param_iteration_poly_ids:
        for pix in db.get_simulations_by_poly(poly_id)[:PIXELS_PER_POLYGON]:
            pix_metadata = metadata_df.loc[pix.index]
            event_date_lower_bound = pd.Timestamp(pix_metadata["start"]) - pd.Timedelta(weeks=slack_weeks_for_event_date)
            event_date_upper_bound = pd.Timestamp(pix_metadata["end"]) + pd.Timedelta(weeks=slack_weeks_for_event_date)

            rmse = root_mean_squared_error(pix.true_signal.to_numpy(), pix.pred_signal.to_numpy())
            upper_bound = pix.pred_signal + k * rmse
            lower_bound = pix.pred_signal - k * rmse

            detected_changes = detect_changes_in_pixel(pix, N, k, majority_threshold).detected_changes

            fig, ax = plt.subplots()
            pred_plot, = ax.plot(pix.pred_signal, "o", markersize=2, label="ESN pred")
            ax.plot(pix.true_signal, "o", markersize=2, label="Denoised signal")
            ax.fill_between(
                pix.pred_signal.index, lower_bound, upper_bound,
                color=pred_plot.get_color(), alpha=0.2, label="MSE region",
            )
            ax.axvline(x=event_date_lower_bound, color="r", linestyle="--", label="LB")
            ax.axvline(x=event_date_upper_bound, color="r", linestyle="--", label="UB")
            for i, change_date in enumerate(detected_changes[detected_changes == 1].index):
                ax.axvline(x=change_date, color="m", linestyle=":", label="Detected change" if i == 0 else None)

            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend(loc="lower right")
            ax.set_title(f"{pix_metadata['change_type']} - Poly {pix.index[0]} - Pixel {pix.index[1]}")
            plt.xticks(rotation=45)
            fig.savefig(
                paths.reports_figures_dir(f"pixel_detection_{selected_band}_poly_{pix.index[0]}_pixel_{pix.index[1]}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)


if __name__ == "__main__":
    plot_pixel_detections()
