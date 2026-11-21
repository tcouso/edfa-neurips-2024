import yaml
import pandas as pd

from src import paths
from src.inference.simulation.pixel_simulation import (
    parallel_simulate_pixels,
)


def run_simulations() -> None:

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    esn_features_dim: int = params["esn_features_dim"]
    change_detection_num_processes: int = params["change_detection_num_processes"]
    esn_num_units: int = params["esn_num_units"]

    signal_df = pd.read_csv(
        paths.data_processed_dir(f"test_{selected_band}.csv"),
        index_col=["ID", "IDpix"],
    )
    signal_df.columns = pd.to_datetime(signal_df.columns)

    model_filename = "_".join(
        ["trained_esn", selected_band, str(esn_features_dim), "features", str(esn_num_units), "units"]
    )
    model_filename += ".pk"

    parallel_simulate_pixels(
        model_path=paths.models_dir(model_filename),
        signal_df=signal_df,
        num_features=esn_features_dim,
        num_processes=change_detection_num_processes,
    )
