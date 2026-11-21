import yaml
import pickle
import pandas as pd

from src import paths
from src.inference.utils.pandas_wrapper import PandasWrapper
from src.inference.change_detection.change_detection import (
    parallel_detect_changes_in_pixels,
)
from src.inference.simulation.simulation_data_structures import get_db_instance


def run_change_detection_param_iter(selected_change_type: str):

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    change_detection_num_processes = params["change_detection_num_processes"]
    param_iteration_poly_ids = params["param_iteration_poly_ids"]
    esn_features_dim = params["esn_features_dim"]
    majority_threshold = params["majority_threshold"]

    simulations_db = get_db_instance()

    selected_pixels = []

    for poly_id in param_iteration_poly_ids:
        selected_pixels.extend(simulations_db.get_simulations_by_poly(poly_id))

    selected_pixels_dict = dict((pix.index, pix) for pix in selected_pixels)
    simulations_df = PandasWrapper(selected_pixels_dict, object_name="simulation")

    metadata_df = pd.read_csv(
        paths.data_processed_dir(f"metadata_annotated_filtered.csv"),
        index_col=["ID", "IDpix"],
        )
    
    # Filter simulations
    selected_polys_metadata_df = metadata_df.loc[param_iteration_poly_ids].copy()

    if selected_change_type != "all":
        excluded_change_type = "fire" if selected_change_type == "logging" else "logging"
        selected_indices = selected_polys_metadata_df.loc[selected_polys_metadata_df["change_type"] != excluded_change_type].index
        selected_simulations_df = simulations_df.loc[selected_indices]
    else:
        selected_simulations_df = simulations_df

    results_filename = f"param_iteration_change_detections_{selected_band}_{esn_features_dim}_features_{selected_change_type}.pk"
    N_values = params["N_values"]
    k_values = params["k_values"]

    results = parallel_detect_changes_in_pixels(
        selected_simulations_df,
        N_values=N_values,
        k_values=k_values,
        majority_threshold=majority_threshold,
        num_processes=change_detection_num_processes,
    )

    change_detections_by_params = {}

    for result in results:
        params_index = (result.N, result.k)
        pixel_index = result.index

        if params_index not in change_detections_by_params:
            change_detections_by_params[params_index] = {}

        change_detections_by_params[params_index][pixel_index] = result

    with open(paths.results_change_detections_dir(results_filename), "wb") as file:
        pickle.dump(change_detections_by_params, file)
