"""
This module provides tools for detecting changes in pixel NDVI simulations using parallel processing.

Classes:
    PixelChangeDetection: Stores detected changes for a single pixel.

Functions:
    detect_changes_in_pixel(pixel_simulation, N=10, k=0.8, offset=1, forecasted_steps=52) -> List[pd.Timestamp]:
        Detects changes in a pixel simulation by comparing forecasted values with actual values.
    detect_changes_in_pixel_caller(args):
        Helper function to unpack arguments for detect_changes_in_pixel.
    parallel_detect_changes_in_pixels(simulations, num_processes, change_detection_forecasted_steps, N_values, k_values, offset_values) -> List[PixelChangeDetection]:
        Parallel process to detect changes in multiple pixel simulations.
"""

import pandas as pd
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from itertools import product
from sklearn.metrics import root_mean_squared_error

from src.inference.simulation.pixel_simulation import PixelSimulation
from src.inference.utils.pandas_wrapper import PandasWrapper
from src.inference.change_detection.change_detection_data_structures import (
    PixelChangeDetection,
)


def detect_changes_in_pixel(
    pixel_simulation: PixelSimulation,
    N: int,
    k: float,
    majority_threshold: float,
) -> PixelChangeDetection:

    detected_changes = pd.Series(index=pixel_simulation.true_signal.index)

    rmse = root_mean_squared_error(
        pixel_simulation.true_signal.to_numpy(), 
        pixel_simulation.pred_signal.to_numpy()
    )

    for i in range(pixel_simulation.true_signal.shape[0] - N):
        date = pixel_simulation.true_signal.index[i + N]
        
        true_window = pixel_simulation.true_signal.iloc[i : i + N]
        pred_window = pixel_simulation.pred_signal.iloc[i : i + N]

        detection_window = (pred_window - true_window) / (k * rmse)

        proportion_above_threshold = (detection_window > 1).mean()

        if proportion_above_threshold >= majority_threshold:
            detected_changes.loc[date] = 1
        else:
            detected_changes.loc[date] = 0

    # Detect the start of each sequence of consecutive 1s
    processed_changes = detected_changes.where(detected_changes.diff().fillna(1) != 0, 0)

    pixel_change_detection = PixelChangeDetection(
        index=pixel_simulation.index,
        N=N,
        k=k,
        detected_changes=processed_changes,
    )
    return pixel_change_detection


def detect_changes_in_pixel_caller(args):
    """
    Helper function to unpack arguments for detect_changes_in_pixel.

    Parameters
    ----------
    args : tuple
        The arguments to be passed to detect_changes_in_pixel.

    Returns
    -------
    List[pd.Timestamp]
        The result of the detect_changes_in_pixel function.
    """
    pixel_simulation, N, k, majority_threshold = args
    pixel_change_detection = detect_changes_in_pixel(
        pixel_simulation=pixel_simulation,
        N=N,
        k=k,
        majority_threshold=majority_threshold,

    )

    return pixel_change_detection


def generate_args(simulations, N_values, k_values, majority_threshold):
    simulations_indices = simulations.index
    for pix_id, N, k in product(simulations_indices, N_values, k_values):

        yield (simulations.loc[pix_id], N, k, majority_threshold)


def parallel_detect_changes_in_pixels(
    simulations: PandasWrapper,
    num_processes: int,
    N_values: List[int],
    k_values: List[float],
    majority_threshold: float,
) -> List[PixelChangeDetection]:
    """
    Parallel process to detect changes in multiple pixel simulations.

    Parameters
    ----------
    simulations : PandasWrapper
        Wrapper containing the pixel simulations.
    num_processes : int
        Number of parallel processes.
    change_detection_forecasted_steps : int
        Number of steps in each forecast.
    N_values : List[int]
        List of detection window lengths.
    k_values : List[float]
        List of threshold factors for detecting significant changes.
    offset_values : List[int]
        List of offsets between the actual signal and the predicted signal to compare.

    Returns
    -------
    List[PixelChangeDetection]
        List of PixelChangeDetection objects for each pixel.
    """
    args_generator = generate_args(simulations, N_values, k_values, majority_threshold)

    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(detect_changes_in_pixel_caller, args)
            for args in args_generator
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Detecting changes"
        ):
            results.append(future.result())

    return results
