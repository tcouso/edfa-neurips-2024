import pandas as pd
from typing import List, Callable


def resample_time_series(time_series: pd.Series, sample_period: int) -> pd.Series:
    """
    Description:
    This function takes a pandas Series `time_series` as input and
    downsamples it by month using the median of the data within each month.

    Parameters:
    - `time_series` (Series): The input pandas Series containing the time series data.

    Returns:
    - Series: The resampled time series.
    """
    # Resample by month
    resampled_time_series = time_series.resample(pd.Timedelta(days=sample_period)).median()

    # Fill missing values by interpolation
    resampled_time_series = resampled_time_series.interpolate()

    # Fill initial missing values by backward fill
    resampled_time_series = resampled_time_series.bfill()

    return resampled_time_series


def moving_std_filter(
    time_series: pd.Series, sample_period: int, window_size: int = 12, std_range: int = 4, 
) -> pd.Series:
    """
    Description:
    This function takes a pandas Series `time_series` as input and filters
    the data based on the rolling mean and standard deviation
    within a specified window.

    Parameters:
    - `time_series` (Series): The input pandas Series containing the time series data.
    - `window_size` (int): The size of the window used to compute the rolling mean and standard deviation (default: 26).
    - `std_range` (int): The number of standard deviations used to calculate the upper and lower bounds (default: 4).

    Returns:
    - Series: The filtered time series.

    """
    # Compute rolling metrics
    rolling_mean = time_series.rolling(window=window_size).mean()
    rolling_std = time_series.rolling(window=window_size).std()

    # Compute thresholds
    rolling_threshold_ub = rolling_mean + (std_range / 2) * rolling_std
    rolling_threshold_lb = rolling_mean - (std_range / 2) * rolling_std

    # Filter out of bounds values
    filtered_time_series = time_series[
        (time_series < rolling_threshold_ub) & (time_series > rolling_threshold_lb)
    ]

    # Interpolate missing values
    filtered_time_series = filtered_time_series.resample(pd.Timedelta(days=sample_period)).median().interpolate()

    # Fill initial missing values by backward fill
    filtered_time_series = filtered_time_series.bfill()

    return filtered_time_series


def denoise_signal(time_series: pd.Series, processors: List[Callable], sample_period: int=16) -> pd.Series:
    """
    Description:
    Iterator pattern for denoising procedure.

    Parameters:
    - `time_series` (Series): The input pandas Series containing the time series data.
    - `processors` (List[Callable]): List of functions to apply sequentially to the time series data.

    Returns:
    - Series: The processed time series.
    """
    denoised_ts = time_series.copy()
    for processor in processors:
        denoised_ts = processor(denoised_ts, sample_period=sample_period)

    return denoised_ts
