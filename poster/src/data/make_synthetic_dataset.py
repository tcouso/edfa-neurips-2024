import json
import numpy as np
import pandas as pd
import yaml
from src import paths
from src.data.utils import create_output_path

POLYGONS_PER_EVENT_TYPE = {"stable": 5, "drought": 5, "logging": 3, "fire": 3}
PIXELS_PER_POLYGON = 3
FIRST_DATE = "2000-01-01"
LAST_DATE = "2022-06-30"
SAMPLE_PERIOD = "16D"
MISSING_RATE = 0.15
OUTLIER_RATE = 0.01
NOISE_STD = 0.03
EVENT_DURATION_DAYS = 540
SEED = 0


def make_pixel_signal(event_type, dates, change_start, rng):
    day_of_year = dates.dayofyear.to_numpy()
    seasonal = 0.6 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    signal = seasonal + rng.normal(0, NOISE_STD, len(dates))

    if event_type == "logging":
        in_event = (dates >= change_start) & (dates < change_start + pd.Timedelta(days=EVENT_DURATION_DAYS))
        signal[in_event] = 0.2 + rng.normal(0, NOISE_STD, in_event.sum())
    elif event_type == "fire":
        after_change = dates >= change_start
        years_since = (dates[after_change] - change_start).days.to_numpy() / 365.25
        recovery = 1 - np.exp(-years_since)
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
    training_limit_year = pd.Timestamp(params["training_limit_date"]).year
    param_iteration_poly_ids = params["param_iteration_poly_ids"]

    rng = np.random.default_rng(SEED)
    dates = pd.date_range(FIRST_DATE, LAST_DATE, freq=SAMPLE_PERIOD)

    # The polygons used for parameter iteration must exist, one of each event type
    polygon_ids = {event_type: [] for event_type in POLYGONS_PER_EVENT_TYPE}
    for event_type, poly_id in zip(POLYGONS_PER_EVENT_TYPE, param_iteration_poly_ids):
        polygon_ids[event_type].append(poly_id)
    next_id = 1
    for event_type, num_polygons in POLYGONS_PER_EVENT_TYPE.items():
        while len(polygon_ids[event_type]) < num_polygons:
            if next_id not in param_iteration_poly_ids:
                polygon_ids[event_type].append(next_id)
            next_id += 1

    metadata_rows = []
    signal_rows = []
    annotation_rows = []
    pixel_id = 0

    for event_type, poly_ids in polygon_ids.items():
        for poly_id in poly_ids:
            has_change = event_type in ("logging", "fire")
            if has_change:
                year = rng.integers(training_limit_year + 3, 2020)
                month = rng.integers(1, 13)
                change_start = pd.Timestamp(year=year, month=month, day=15)
                change_end = change_start + pd.Timedelta(days=EVENT_DURATION_DAYS)
                annotation_rows.append({
                    "csv": f"ID_{poly_id}.csv",
                    "label": json.dumps([{
                        "start": change_start.strftime("%Y-%m-%d"),
                        "end": change_end.strftime("%Y-%m-%d"),
                    }]),
                })
            else:
                change_start = None

            for _ in range(PIXELS_PER_POLYGON):
                pixel_id += 1
                signal = make_pixel_signal(event_type, dates, change_start, rng)
                signal_rows.append(
                    {"ID": poly_id, "IDpix": pixel_id,
                     **{date.strftime("%Y-%m-%d"): value for date, value in zip(dates, signal)}}
                )
                metadata_rows.append({
                    "ID": poly_id,
                    "IDpix": pixel_id,
                    "change_type": event_type,
                    "label": int(has_change),
                })

    outputs = [
        (paths.data_raw_dir("metadata.csv"), pd.DataFrame(metadata_rows)),
        (paths.data_raw_dir(selected_band + ".csv"), pd.DataFrame(signal_rows)),
        (paths.data_external_dir("change_poly_annotations.csv"), pd.DataFrame(annotation_rows)),
    ]
    for path, df in outputs:
        create_output_path(path)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    make_synthetic_dataset()
