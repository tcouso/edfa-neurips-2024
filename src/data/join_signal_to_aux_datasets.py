import pandas as pd
import yaml
from pathlib import Path
from src import paths
from typing import List
from src.data.utils import create_output_path


def join_signal_to_aux_datasets(
    params_path: Path = paths.config_dir("params.yaml"),
) -> None:

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    event_types: List[str] = params["event_types"]
    test_at_scale: bool = params["test_at_scale"]
    test_at_scale_sample_size: int = params["test_at_scale_sample_size"]

    for event_type in event_types:
        signal_filepath = paths.data_raw_dir(
            event_type, "signal_" + selected_band + ".csv")
        metadata_filepath = paths.data_raw_dir(event_type, "metadata.csv")

        signal_df = pd.read_csv(signal_filepath)
        metadata_df = pd.read_csv(metadata_filepath)

        df = pd.merge(metadata_df, signal_df, on="IDpix", how="outer")
        df = df.set_index(["ID", "IDpix"])
        df["change_type"] = event_type

        out_path = paths.data_interim_dir(
            event_type, event_type + "_" + selected_band + ".csv")

        create_output_path(out_path)

        if test_at_scale:
            df = df.sample(n=test_at_scale_sample_size)

        df.to_csv(out_path)
