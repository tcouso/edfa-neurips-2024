import numpy as np
import pandas as pd
import yaml
from src import paths
from src.data.utils import create_output_path

POLYGONS_PER_EVENT_TYPE = {"stable": 6, "drought": 6, "logging": 4, "fire": 4}
PIXELS_PER_POLYGON = 5
FIRST_DATE = "2000-01-01"
LAST_DATE = "2022-06-30"
SAMPLE_PERIOD = "16D"
MISSING_RATE = 0.15
OUTLIER_RATE = 0.01
NOISE_STD = 0.03
SEED = 0


def make_pixel_signal(event_type, dates, change_start, rng):
    day_of_year = dates.dayofyear.to_numpy()
    seasonal = 0.6 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    signal = seasonal + rng.normal(0, NOISE_STD, len(dates))

    after_change = dates >= change_start if change_start is not None else None
    if event_type == "logging":
        signal[after_change] = 0.2 + rng.normal(0, NOISE_STD, after_change.sum())
    elif event_type == "fire":
        years_since = (dates[after_change] - change_start).days.to_numpy() / 365.25
        recovery = 1 - np.exp(-years_since / 3)
        signal[after_change] = 0.2 + 0.4 * recovery * (seasonal[after_change] / 0.6)
        signal[after_change] += rng.normal(0, NOISE_STD, after_change.sum())
    elif event_type == "drought":
        in_drought = (dates >= "2019-06-01") & (dates <= "2020-12-31")
        signal[in_drought] -= 0.1

    signal[rng.random(len(dates)) < MISSING_RATE] = np.nan
    outliers = rng.random(len(dates)) < OUTLIER_RATE
    signal[outliers] = rng.uniform(-0.2, 0.2, outliers.sum())
    return signal


def make_synthetic_dataset() -> None:
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    event_threshold_year = pd.Timestamp(params["event_threshold"]).year

    rng = np.random.default_rng(SEED)
    dates = pd.date_range(FIRST_DATE, LAST_DATE, freq=SAMPLE_PERIOD)

    polygon_id = 0
    pixel_id = 0

    for event_type, num_polygons in POLYGONS_PER_EVENT_TYPE.items():
        metadata_rows = []
        signal_rows = []

        for _ in range(num_polygons):
            polygon_id += 1
            has_change_date = event_type in ("logging", "fire")
            change_start = None
            if has_change_date:
                year = rng.integers(event_threshold_year + 1, 2020)
                month = rng.integers(1, 13)
                change_start = pd.Timestamp(year=year, month=month, day=15)

            for _ in range(PIXELS_PER_POLYGON):
                pixel_id += 1
                signal = make_pixel_signal(event_type, dates, change_start, rng)
                signal_rows.append(
                    {"IDpix": pixel_id,
                     **{date.strftime("%Y-%m-%d"): value for date, value in zip(dates, signal)}}
                )
                metadata_rows.append({
                    "ID": polygon_id,
                    "IDpix": pixel_id,
                    "lat": -34.0 + rng.normal(0, 0.1),
                    "lon": -71.0 + rng.normal(0, 0.1),
                    "change_start": change_start.strftime("%d-%m-%Y") if has_change_date else "",
                    "last_non_change_date": (change_start - pd.Timedelta(days=30)).strftime("%d-%m-%Y") if has_change_date else "",
                    "vegetation_type": "native",
                })

        signal_path = paths.data_raw_dir(event_type, "signal_" + selected_band + ".csv")
        metadata_path = paths.data_raw_dir(event_type, "metadata.csv")
        create_output_path(signal_path)
        create_output_path(metadata_path)
        pd.DataFrame(signal_rows).to_csv(signal_path, index=False)
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)


if __name__ == "__main__":
    make_synthetic_dataset()
