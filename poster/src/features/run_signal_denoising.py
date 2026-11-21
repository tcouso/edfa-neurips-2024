import pandas as pd
from pathlib import Path
import yaml

from src.features.denoise_signal import (
    denoise_signal,
    resample_time_series,
    moving_std_filter,
)
from src import paths


def run_signal_denoising() -> None:

    params_path: Path = paths.config_dir("params.yaml")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]

    train_signal_filename = "_".join(["train", selected_band])
    train_signal_filename += ".csv"

    signal_df = pd.read_csv(
        paths.data_processed_dir(train_signal_filename), index_col=["ID", "IDpix"]
    )
    signal_df.columns = pd.to_datetime(signal_df.columns)

    denoised_train_signal = []
    max_start_date = pd.Timestamp.min
    min_end_date = pd.Timestamp.max

    for pix_index in signal_df.index:
        denoised_ts = denoise_signal(
            signal_df.loc[pix_index],
            [
                resample_time_series,
                moving_std_filter,
            ],
        )
        denoised_train_signal.append(denoised_ts)

        # Identify max start date
        if denoised_ts.index.min() > max_start_date:
            max_start_date = denoised_ts.index.min()

        # Identify min end date
        if denoised_ts.index.max() < min_end_date:
            min_end_date = denoised_ts.index.max()

    for i, _ in enumerate(signal_df.index):
        # Adjust polygon dates for equal size
        denoised_train_signal[i] = denoised_train_signal[i][
            max_start_date:
        ]
        denoised_train_signal[i] = denoised_train_signal[i][
            :min_end_date
        ]

    denoised_train_signal_df = pd.DataFrame(denoised_train_signal)
    denoised_train_signal_df.index = signal_df.index

    denoised_train_signal_filename = "_".join(["denoised", train_signal_filename])
    denoised_train_signal_df.to_csv(paths.features_dir(denoised_train_signal_filename))
