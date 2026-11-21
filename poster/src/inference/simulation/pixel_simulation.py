"""
This module provides tools for simulating NDVI dynamics for individual pixels using a trained ESN model,
as well as parallel processing capabilities for efficiency.

Classes:
    PixelSimulation: Stores and manages the simulation of NDVI dynamics for a single pixel.
    PixelSimulationDB: Manages storage and retrieval of PixelSimulation objects using SQLite.

Functions:
    simulate_pixel(model, pixel_index, signal, num_features=104, forecasted_steps=52) -> PixelSimulation:
        Simulates the future signal dynamics for a pixel using a trained ESN model.
    simulate_pixel_caller(args):
        Helper function to unpack arguments for simulate_pixel.
    parallel_simulate_pixels(model_path, pixel_simulation_db, signal_df, num_features=104, forecasted_steps=52, num_processes=4) -> None:
        Parallel process each pixel for NDVI forecasting using a pool of workers.
"""


from typing import List, Tuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from tqdm import tqdm
import reservoirpy as rpy

from src.features.slice_time_series import create_training_data
from src.features.denoise_signal import (
    resample_time_series,
    moving_std_filter,
    denoise_signal,
)
from src.inference.simulation.simulation_data_structures import (
    PixelSimulation,
    get_db_instance,
)

rpy.verbosity(0)


def simulate_pixel(
    model: object,
    pixel_index: Tuple[int, int],
    signal: pd.Series,
    num_features: int,
) -> PixelSimulation:
    """
    Simulates the future signal dynamics for a pixel using a trained ESN model.

    Parameters
    ----------
    model : object
        The trained ESN model.
    pixel_index : Tuple[int, int]
        The (row, column) index of the pixel in the image.
    signal : pd.Series
        The actual NDVI signal for the pixel.
    num_features : int, optional
        Number of features used for training, by default 104.
    forecasted_steps : int, optional
        Number of steps to forecast, by default 52.

    Returns
    -------
    PixelSimulation
        An object containing the simulation results for the pixel.
    """

    denoised_signal_series = denoise_signal(
        signal,
        [
            resample_time_series,
            moving_std_filter,
        ],
    )

    denoised_signal_dates = denoised_signal_series.index.to_numpy().reshape(-1, 1)
    pred_dates = denoised_signal_dates[num_features:, :]

    denoised_signal = denoised_signal_series.to_numpy()
    X, y = create_training_data(denoised_signal, num_features=num_features)

    pred_signal = model.run(X)

    pred_series = pd.Series(pred_signal.flatten(), index=pred_dates.flatten())
    true_series = pd.Series(y.flatten(), index=pred_dates.flatten())

    pixel_simulation = PixelSimulation(
        index=pixel_index,
        raw_signal=signal,
        true_signal=true_series,
        pred_signal=pred_series,
    )

    return pixel_simulation


def simulate_pixel_caller(args):
    """
    Helper function to unpack arguments for simulate_pixel.

    Parameters
    ----------
    args : tuple
        The arguments to be passed to simulate_pixel.

    Returns
    -------
    PixelSimulation
        The result of the simulate_pixel function.
    """
    model, pixel_index, signal, num_features = args
    return simulate_pixel(model, pixel_index, signal, num_features)


def generate_simulation_args(pretrained_model, signal_df, num_features, pixel_indices):
    for idx in pixel_indices:
        yield (pretrained_model, idx, signal_df.loc[idx], num_features)


def parallel_simulate_pixels(
    model_path: str,
    signal_df: pd.DataFrame,
    num_features: int,
    num_processes: int,
) -> None:
    """
    Parallel process each pixel for NDVI forecasting using a pool of workers.

    Parameters
    ----------
    model_path : str
        Path to the pretrained model pickle file.
    signal_df : pd.DataFrame
        DataFrame containing the NDVI signals.
    num_features : int, optional
        Number of features used for training, by default 104.
    forecasted_steps : int, optional
        Number of steps to forecast, by default 52.
    num_processes : int, optional
        Number of parallel processes, by default 4.

    Returns
    -------
    List[PixelSimulation]
        List of PixelSimulation objects for each pixel.
    """
    with open(model_path, "rb") as file:
        pretrained_model = pickle.load(file)

    pixel_simulation_db = get_db_instance()

    pixel_indices = signal_df.index

    args_generator = generate_simulation_args(
        pretrained_model, signal_df, num_features, pixel_indices
    )
    simulations = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(simulate_pixel_caller, args) for args in args_generator
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Simulating pixels"
        ):
            simulation = future.result()
            simulations.append(simulation)

    for simulation in tqdm(simulations, desc="Storing simulations in database"):
        pixel_simulation_db.insert_simulation(simulation)


def parallel_simulate_pixels_production(
    model_path: str,
    signal_df: pd.DataFrame,
    num_features: int,
    num_processes: int,
) -> List[PixelSimulation]:
    """
    Parallel simulation of each pixel, returning the simulations instead of storing them in the database.
    """
    with open(model_path, "rb") as file:
        pretrained_model = pickle.load(file)

    args_generator = generate_simulation_args(
        pretrained_model, signal_df, num_features, signal_df.index
    )
    simulations = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(simulate_pixel_caller, args) for args in args_generator
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Simulating pixels"
        ):
            simulations.append(future.result())

    return simulations
