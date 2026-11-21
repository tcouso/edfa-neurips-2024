import yaml
import pandas as pd
import rioxarray as riox
from src import paths
from src.inference.simulation.pixel_simulation import parallel_simulate_pixels_production
from src.inference.change_detection.change_detection import parallel_detect_changes_in_pixels
from src.inference.utils.pandas_wrapper import PandasWrapper
from src.config import get_logger


def detect_over_raster(img, dates, detection_start, reference_date, num_workers, band_dim):
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    esn_features_dim = params["esn_features_dim"]
    esn_num_units = params["esn_num_units"]
    N = params["optim_N_all"]
    k = params["optim_k_all"]
    majority_threshold = params["majority_threshold"]

    img_flat = img.stack(z=("y", "x"))
    img_flat = img_flat.transpose("z", "band")
    signal_df = pd.DataFrame(img_flat.values)

    signal_df.index = pd.MultiIndex.from_tuples(
        [(0, idx) for idx in signal_df.index], names=["ID", "IDpix"]
    )
    signal_df.columns = dates
    signal_df.columns.name = "date"

    model_filename = "_".join(
        ["trained_esn", selected_band, str(esn_features_dim), "features", str(esn_num_units), "units"]
    )
    model_filename += ".pk"
    simulations = parallel_simulate_pixels_production(
        model_path=paths.models_dir(model_filename),
        signal_df=signal_df,
        num_features=esn_features_dim,
        num_processes=num_workers,
    )
    simulations_dict = dict((pix.index, pix) for pix in simulations)
    simulations_df = PandasWrapper(simulations_dict, object_name="simulation")

    change_detections = parallel_detect_changes_in_pixels(
        simulations_df,
        num_processes=num_workers,
        majority_threshold=majority_threshold,
        N_values=[N],
        k_values=[k],
    )
    change_detections_dict = dict((pix.index, pix) for pix in change_detections)
    change_detections_df = PandasWrapper(change_detections_dict, object_name="change_detections")

    detections_series = pd.Series(index=change_detections_df.index, dtype=int)
    event_dates_series = pd.Series(index=change_detections_df.index, dtype="datetime64[ns]")

    for idx in change_detections_df.index:
        detected_changes = change_detections_df.loc[idx].detected_changes[detection_start:]

        if any(detected_changes == 1):
            detections_series[idx] = 1
            event_dates_series[idx] = pd.Series(detected_changes[detected_changes == 1].index).median()
        else:
            detections_series[idx] = 0
            event_dates_series[idx] = pd.NaT

    template = img_flat[:, band_dim]
    detections_img = template.copy(data=detections_series.to_numpy()).unstack()

    event_dates_encoded = event_dates_series.apply(
        lambda x: (x - reference_date).days if pd.notna(x) else -1
    )
    detection_dates_img = template.copy(data=event_dates_encoded.to_numpy()).unstack()
    detection_dates_img.attrs.update({"reference_date": reference_date})

    return detections_img, detection_dates_img


def detect_over_rasters() -> None:
    logger = get_logger()

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    num_workers = params["change_detection_num_processes"]
    detection_start = pd.Timestamp(params["raster_detection_start_date"])
    reference_date = pd.Timestamp(params["raster_reference_date"])

    rasters_dir = paths.data_external_dir("rasters")
    dates_dir = paths.data_external_dir("raster_dates")
    out_dir = paths.data_processed_dir("raster_detections")
    out_dir.mkdir(parents=True, exist_ok=True)

    for raster_path in sorted(rasters_dir.glob("*.tif")):
        logger.info(f"Detecting changes over {raster_path.name}")

        img = riox.open_rasterio(raster_path).squeeze(drop=True)
        dates = pd.read_csv(dates_dir / raster_path.with_suffix(".csv").name)
        dates = pd.to_datetime(dates["Formatted Date"], format="%Y.%m.%d")

        detections_img, detection_dates_img = detect_over_raster(
            img, dates, detection_start, reference_date, num_workers, band_dim=1
        )

        detections_img.rio.to_raster(out_dir / raster_path.name)
        detection_dates_img.rio.to_raster(out_dir / raster_path.name.replace(".tif", "_detection_dates.tif"))


if __name__ == "__main__":
    detect_over_rasters()
