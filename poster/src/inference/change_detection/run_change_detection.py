import yaml
import pickle
import pandas as pd
import gc

from src import paths
from src.inference.utils.pandas_wrapper import PandasWrapper
from src.inference.change_detection.change_detection import (
    parallel_detect_changes_in_pixels,
)
from src.inference.simulation.simulation_data_structures import get_db_instance
from src.config import get_logger


def run_change_detection():

    logger = get_logger()

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    change_detection_num_processes = params["change_detection_num_processes"]
    esn_features_dim = params["esn_features_dim"]
    poly_ids_to_exclude = params["param_iteration_poly_ids"]
    N_values = [params["optim_N"]]
    k_values = [params["optim_k"]]
    majority_threshold = params["majority_threshold"]

    metadata_df = pd.read_csv(
        paths.data_processed_dir(f"metadata_annotated_filtered.csv"),
        index_col=["ID", "IDpix"],
    )
    poly_ids = metadata_df.index.get_level_values(0).unique()
    simulations_db = get_db_instance()

    for poly_id in poly_ids:
        if poly_id in poly_ids_to_exclude:
            logger.info(f"Excluding polygon {poly_id}")
            continue
        logger.info(f"Detecting changes in polygon {poly_id}")
        selected_pixels = simulations_db.get_simulations_by_poly(poly_id)
        selected_pixels_dict = dict((pix.index, pix) for pix in selected_pixels)
        simulations_df = PandasWrapper(selected_pixels_dict, object_name="simulation")

        poly_resuls = parallel_detect_changes_in_pixels(
            simulations_df,
            num_processes=change_detection_num_processes,
            majority_threshold=majority_threshold,
            N_values=N_values,
            k_values=k_values,
        )

        poly_change_detections_by_params = {}

        for result in poly_resuls:
            params_index = (result.N, result.k)
            pixel_index = result.index

            if params_index not in poly_change_detections_by_params:
                poly_change_detections_by_params[params_index] = {}

            poly_change_detections_by_params[params_index][pixel_index] = result

        results_filename = f"change_detections_{selected_band}_poly_{poly_id}_pixels_{esn_features_dim}_features.pk"
        results_path = paths.results_optimal_change_detections_dir(results_filename)

        with open(results_path, "wb") as file:
            pickle.dump(poly_change_detections_by_params, file)

        # Clean up to free memory
        del (
            selected_pixels,
            selected_pixels_dict,
            simulations_df,
            poly_resuls,
            poly_change_detections_by_params,
        )
        gc.collect()
