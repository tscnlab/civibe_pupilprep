import pandas as pd
import numpy as np
import os
import pupilprep_utilities.loading_utils as load
import pupilprep_utilities.preprocessing_utils as prep


def load_and_resample(
    raw_data_dir: str,
    participant_id: int,
    include_failed_raw: bool = False,
    save_raw: bool = True,
    save_path_raw: str = "./results/raw/",
    delay_raw: float = 0,
    resampling_frequency: int = 30,
    **kwargs
):
    """Pipeline for loading and resampling data.
    For more details on functions/variables, refer to load.load_participant_data and prep.resample_by_trial.

    Args:
        raw_data_dir (str): path to raw data.
        participant_id (int): participant number.
        include_failed_raw (bool, optional): whether to include failed runs. Defaults to False.
        save_raw (bool, optional): whether to save raw data with trial defining variables added. Defaults to True.
        save_path_raw (str, optional): save path for raw data. Defaults to "./results/raw/".
        delay_raw (float, optional): SW-HW delay of raw data. Defaults to 0.
        resampling_frequency (int, optional): Frequency to resample to. Defaults to 30.

    Returns:
        pd.DataFrame: resampled dataframe
    """

    data_df, protocol_timecourse_df, protocol_vars_df = load.load_participant_data(
        participant_no=participant_id,
        data_dir=raw_data_dir,
        delay=delay_raw,
        include_failed=include_failed_raw,
        save=save_raw,
        save_path=save_path_raw,
    )

    data_df = prep.remove_artefacts_non_physio_size(data_df)
    data_df = prep.resample_by_trial(data_df, sample_freq=resampling_frequency)

    return data_df


def remove_artefacts(
    data_df: pd.DataFrame,
    rolling_window_velocity: int = 60,
    rolling_window_size: int = 60,
    multiplier_velocity: float = 6,
    multiplier_size: float = 4.5,
    columns: list = ["Stim eye - Size Mm"],
    **kwargs
):
    """Pipeline for removing artefacts from resampled data. For more details, refer to prep.remove_artefacts_... functions.

    Args:
        data_df (pd.DataFrame): resampled data.
        rolling_window_velocity (int, optional): window size for velocity MAD in samples. Defaults to 60.
        rolling_window_size (int, optional): window size for size MAD in samples. Defaults to 60.
        multiplier_velocity (float, optional): multiplier for velocity MAD. Defaults to 6.
        multiplier_size (float, optional): multiplier for size MAD. Defaults to 4.5.
        columns (list, optional): list of strings - column names to remove artefacts from. Defaults to [ "Stim eye - Size Mm" ].

    Returns:
        pd.DataFrame: dataframe with artefacts in specified columns removed
    """
    data_df = data_df.copy()

    for column in columns:
        data_df = prep.remove_artefacts_rolling_velocity_mad(
            resampled_df=data_df,
            multiplier=multiplier_velocity,
            window=rolling_window_velocity,
            column=column,
        )
        data_df = prep.remove_artefacts_rolling_size_mad(
            resampled_df=data_df,
            multiplier=multiplier_size,
            window=rolling_window_size,
            column=column,
        )
    return data_df


def reject_incomplete_data(
    data_df: pd.DataFrame,
    resampling_frequency: int = 30,
    baseline_time: list = [-1, 0],
    poi_time: list = [0, 6],
    baseline_threshold: int = 40,
    poi_threshold: int = 75,
    max_nan_length: int = 625,
    trial_min: int = 3,
    **kwargs
):
    """Pipeline for removal of incomplete data. For details, refer to prep.remove_trials_..., prep.remove_bad_conditions, prep.remove_bad_blocks.

    Args:
        data_df (pd.DataFrame): resampled data with artefacts removed.
        resampling_frequency (int, optional): frequency data was resampled to. Defaults to 30.
        baseline_time (list, optional): [start,end] of baseline period in seconds. Defaults to [-1, 0].
        poi_time (list, optional): [start,end] of period of interest in seconds. Defaults to [0, 6].
        baseline_threshold (int, optional): completeness percentage threshold for baseline. Defaults to 40.
        poi_threshold (int, optional): completeness percentage threshold for period of interest. Defaults to 75.
        max_nan_length (int, optional): maximum length of NaN sequence in POI, in ms. Defaults to 625.
        trial_min (int, optional): minimum number of trials for condition in block. Defaults to 3.

    Returns:
        pd.DataFrame: dataframe with only blocks/conditions fulfilling completeness requirements
    """

    data_df = data_df.copy()

    data_df = prep.remove_trials_below_percentage(
        resampled_df=data_df,
        baseline_threshold=baseline_threshold,
        poi_threshold=poi_threshold,
        baseline_time=baseline_time,
        poi_time=poi_time,
    )

    data_df = prep.remove_trials_with_long_nans(
        thresholded_df=data_df,
        fs=resampling_frequency,
        max_nan_length=max_nan_length,
        poi_time=poi_time,
    )

    data_df = prep.remove_bad_conditions(data_df=data_df, trial_min=trial_min)

    data_df = prep.remove_bad_blocks(data_df=data_df)

    return data_df


def full_preprocessing_pipeline(
    raw_data_dir: str,
    participant_id: int,
    save_path_resampled: str = "./results/resampled/",
    save_path_cleaned: str = "/results/cleaned/",
    save_path_complete: str = "/results/complete/",
    save_intermediate_steps: bool = True,
    **kwargs
):
    """Main pipeline for full preprocessing.
    For details of available keyword arguments (**kwargs), look in the load_and_resample, remove_artefacts, reject_incomplete_data pipelines.

    Args:
        raw_data_dir (str): directory with raw data.
        participant_id (int): participant's number.
        save_path_resampled (str, optional): save path for resampled data. Defaults to './results/resampled/'.
        save_path_cleaned (str, optional): save path for no artefacts data. Defaults to '/results/cleaned/'.
        save_path_complete (str, optional): save path for data after completeness-based rejection. Defaults to '/results/complete/'.
        save_intermediate_steps (bool, optional): whether to save intermediate results of pipeline. Defaults to True.
        kwargs: any keyword arguments you'd like to be changed in the pipeline

    Returns:
        pd.DataFrame: fully preprocessed data from one participant
    """

    data_df = data_df.copy()

    data_df = load_and_resample(raw_data_dir, participant_id, **kwargs)
    if save_intermediate_steps:
        data_df.to_csv(
            os.path.join(
                save_path_resampled, str(participant_id) + "_resampled_data.csv"
            )
        )

    data_df = remove_artefacts(data_df, **kwargs)
    if save_intermediate_steps:
        data_df.to_csv(
            os.path.join(save_path_cleaned, str(participant_id) + "_cleaned_data.csv")
        )

    data_df = reject_incomplete_data(data_df, **kwargs)
    data_df.to_csv(
        os.path.join(save_path_complete, str(participant_id) + "_complete_data.csv")
    )

    return data_df
