import pandas as pd
import numpy as np
import datetime
from abc import ABC

#Functions for resampling data

def resample_by_trial(data_df: pd.DataFrame, sample_freq: int = 30):
    """Function for resampling raw data.

    Args:
        data_df (pd.DataFrame): Dataframe with raw data from one participant from loading_utils.load_participant_data
        sample_freq (int, optional): Desired sampling frequency in Hz to resample data to. Defaults to 30.

    Returns:
        pd.DataFrame: DataFrame resampled to desired frequency with columns truncated to : Trial no, time Sec, time datetime, phase, type, Block, Test, Recording id, Participant id, Eye
    """
    # get time step in ns from sampling frequency provided - ns give greater precision
    time_step = np.ceil((1000 / sample_freq) * 1e6)

    # take subset of data without transition and adaptation parts (so without non-trial values)
    data_subset = data_df[(data_df["Trial no"].notna())]

    # map trial-relevant variables to trial numbers for trial marking after resampling (can't resample strings easily but we know trial data)
    trial_list = sorted(data_subset["Trial no"].unique())
    stim_list = [
        data_subset["Trial type"][data_subset["Trial no"] == i].unique()[0]
        for i in trial_list
    ]
    block_list = [
        data_subset["Block"][data_subset["Trial no"] == i].unique()[0]
        for i in trial_list
    ]
    test_list = [
        data_subset["Test"][data_subset["Trial no"] == i].unique()[0]
        for i in trial_list
    ]
    recording_list = [
        data_subset["Recording id"][data_subset["Trial no"] == i].unique()[0]
        for i in trial_list
    ]
    eye_list = [
        data_subset["Eye"][data_subset["Trial no"] == i].unique()[0] for i in trial_list
    ]
    participant = data_subset["Participant id"].unique()[0]

    # make datetime index for resampling (pandas doesn't work without it)
    data_subset.loc[:,"Trial time datetime"] = data_subset["Trial time Sec"].apply(
        lambda x: datetime.timedelta(seconds=x)
    )
    data_subset.set_index("Trial time datetime", inplace=True)

    # resample by trial and create a new dataframe
    trials_for_new_df = []
    
    for i, trial_no in enumerate(trial_list):
        trial = data_subset[["Trial time Sec", "Stim eye - Size Mm","Right - Size Mm","Left - Size Mm"]][
            data_subset["Trial no"] == trial_no
        ].copy()
        # add a row at -1s so that every trial has the same time ticks
        trial.loc[datetime.timedelta(seconds=-1)] = (
            [pd.NA]*len(trial.columns)
        )  
        # just in case the trial is too short, add row at 18s (this ensures we have all trials at the same length - then they can e.g. easily go into an array)
        trial.loc[datetime.timedelta(seconds=18)] = (
            [pd.NA]*len(trial.columns)
        )  
        resampled_trial = trial.resample(str(time_step) + "ns").agg(
            {"Stim eye - Size Mm": "mean","Right - Size Mm": "mean","Left - Size Mm": "mean"}
        )
        # cut trial to 18 s
        resampled_trial = resampled_trial[
            datetime.timedelta(seconds=-1) : datetime.timedelta(seconds=18)
        ]

        # remake trial time column in seconds from new index
        resampled_trial["Trial time Sec"] = resampled_trial.index
        resampled_trial["Trial time Sec"] = resampled_trial["Trial time Sec"].apply(
            lambda x: x.total_seconds()
        )

        # mark trial based on mappings
        resampled_trial["Trial no"] = [trial_no] * len(resampled_trial)
        resampled_trial["Trial type"] = [stim_list[i]] * len(resampled_trial)
        resampled_trial["Block"] = [block_list[i]] * len(resampled_trial)
        resampled_trial["Test"] = [test_list[i]] * len(resampled_trial)
        resampled_trial["Recording id"] = [recording_list[i]] * len(resampled_trial)
        resampled_trial["Eye"] = [eye_list[i]] * len(resampled_trial)
        resampled_trial["Participant id"] = [participant] * len(resampled_trial)

        # mark trial phases based on protocol
        resampled_trial.loc[resampled_trial["Trial time Sec"] < 0, "Trial phase"] = (
            "pre-stim"
        )
        resampled_trial.loc[
            (resampled_trial["Trial time Sec"] >= 0)
            & (resampled_trial["Trial time Sec"] <= 5),
            "Trial phase",
        ] = "stim"
        resampled_trial.loc[resampled_trial["Trial time Sec"] > 5, "Trial phase"] = (
            "post-stim"
        )
       
        trials_for_new_df.append(resampled_trial)

    new_df = pd.concat(trials_for_new_df)
    new_df.reset_index(inplace=True)
    return new_df


# Functions for rejecting incomplete data

def remove_trials_below_percentage(
    resampled_df: pd.DataFrame,
    baseline_threshold: int = 40,
    poi_threshold: int = 75,
    baseline_time: list = [-1, 0],
    poi_time: list = [0, 6],
):
    """Function for removing trials below data completeness percentage threshold.

    Args:
        resampled_df (pd.DataFrame): resampled dataframe, output of preprocessing_utils.resample_by_trial
        baseline_threshold (int, optional): percentage threshold for baseline. Defaults to 40.
        poi_threshold (int, optional): percentage threshold for period of interest. Defaults to 75.
        baseline_time (list, optional): time borders for baseline in seconds [start, end]. Defaults to [-1, 0].
        poi_time (list, optional): time borders for period of interest in seconds [start, end]. Defaults to [0, 6].

    Returns:
        pd.DataFrame: dataframe with trials not meeting both percentage conditions removed
    """

    resampled_df = resampled_df.copy()

    # compute poi data percentage present in trials
    poi_df = resampled_df[
        (resampled_df["Trial time Sec"] >= poi_time[0])
        & (resampled_df["Trial time Sec"] <= poi_time[1])
    ]
    poi_groupby_df = (
        poi_df[["Trial no", "Stim eye - Size Mm"]]
        .groupby(["Trial no"])
        .agg(["count", "size"])
        .reset_index()
    )
    poi_groupby_df[("Stim eye - Size Mm", "count/size ratio")] = (
        poi_groupby_df[("Stim eye - Size Mm", "count")]
        / poi_groupby_df[("Stim eye - Size Mm", "size")]
    ) * 100

    # compute baseline data percentage present in trials
    baseline_df = resampled_df[
        (resampled_df["Trial time Sec"] >= baseline_time[0])
        & (resampled_df["Trial time Sec"] <= baseline_time[1])
    ]
    baseline_groupby_df = (
        baseline_df[["Trial no", "Stim eye - Size Mm"]]
        .groupby(["Trial no"])
        .agg(["count", "size"])
        .reset_index()
    )
    baseline_groupby_df[("Stim eye - Size Mm", "count/size ratio")] = (
        baseline_groupby_df[("Stim eye - Size Mm", "count")]
        / baseline_groupby_df[("Stim eye - Size Mm", "size")]
    ) * 100

    # find trials matching poi condition and baseline condition
    pois_above_threshold = (
        poi_groupby_df[("Stim eye - Size Mm", "count/size ratio")] >= poi_threshold
    )
    baselines_above_threshold = (
        baseline_groupby_df[("Stim eye - Size Mm", "count/size ratio")]
        >= baseline_threshold
    )
    trials_accepted_indices = pois_above_threshold & baselines_above_threshold
    trials_accepted = poi_groupby_df[("Trial no", "")][trials_accepted_indices]

    # select only found trials from original dataframe
    removed_df = resampled_df[resampled_df["Trial no"].isin(trials_accepted)]
    removed_df = removed_df.reset_index(drop=True)

    return removed_df


def remove_trials_with_long_nans(
    thresholded_df: pd.DataFrame,
    fs: int = 30,
    max_nan_length: int = 625,
    poi_time: list = [0, 6],
):
    """Function for removing trials with NaN sequences exceeding the limit in ms in period of interest.

    Args:
        thresholded_df (pd.DataFrame): dataframe with resampled data.
        fs (int, optional): sampling frequency of dataframe. Defaults to 30.
        max_nan_length (int, optional): maximum NaN sequence length in miliseconds. Defaults to 625.
        poi_time (list, optional): time borders for period of interest in seconds [start,end]. Defaults to [0, 6].

    Returns:
        pd.DataFrame: dataframe with trials exceeding the gap length condition removed
    """
    # select rows in the period of interest
    data_df = thresholded_df[
        (thresholded_df["Trial time Sec"] >= poi_time[0])
        & (thresholded_df["Trial time Sec"] <= poi_time[1])
    ].copy()

    # mark NaN sequences in a counter (e.g. for sequence: 7,NaN,NaN,NaN,5 the counter is: 0,1,2,3,0)
    data_df["NaN counter"] = pd.Series()

    for trial_no in sorted(data_df["Trial no"].unique()):
        trial = data_df[data_df["Trial no"] == trial_no]
        trial_nan_counter = (
            trial["Stim eye - Size Mm"]
            .isnull()
            .astype(int)
            .groupby(trial["Stim eye - Size Mm"].notnull().astype(int).cumsum())
            .cumsum()
        )
        data_df.loc[data_df["Trial no"] == trial_no, "NaN counter"] = trial_nan_counter

    # find trials in which counter exceeds max nan length in samples
    trials_above_max = data_df["Trial no"][
        data_df["NaN counter"] > (max_nan_length / fs)
    ].unique()

    # select trials without sequences exceeding the limit
    removed_df = thresholded_df[~thresholded_df["Trial no"].isin(trials_above_max)]
    removed_df = removed_df.reset_index(drop=True)
    return removed_df


def remove_bad_conditions(data_df: pd.DataFrame, trial_min: int = 3):
    """Function for removing conditions with insufficient trial numbers in a block.

    Args:
        data_df (pd.DataFrame): dataframe with trials removed based on data completeness (preprocessing_utils.remove_trials_below_percentage,remove_trials_with_long_nans).
        trial_min (int, optional): minimum number of trials. Defaults to 3.

    Returns:
        pd.DataFrame: dataframe with conditions removed from blocks where they don't meet the minimum trial number
    """
    # aggregate unique trial numbers in each block-condition group
    groupby_condition_df = (
        data_df[["Block", "Trial type", "Trial no"]]
        .groupby(["Block", "Trial type"])
        .agg({"Trial no": "nunique"})
    )
    groupby_condition_df.reset_index(inplace=True)

    # find block-condition pairs where a condition has less than 3 trials
    low_cond_block_pairs = [
        (block, cond)
        for (block, cond) in zip(
            groupby_condition_df["Block"], groupby_condition_df["Trial type"]
        )
        if groupby_condition_df["Trial no"][
            (groupby_condition_df["Block"] == block)
            & (groupby_condition_df["Trial type"] == cond)
        ].values
        < trial_min
    ]
    # find trial numbers corresponding to the pairs above
    low_cond_trials = [
        trial_no
        for block, cond in low_cond_block_pairs
        for trial_no in data_df["Trial no"][
            (data_df["Block"] == block) & (data_df["Trial type"] == cond)
        ]
    ]

    # remove the found trials from dataframe
    removed_df = data_df[~data_df["Trial no"].isin(low_cond_trials)]
    removed_df = removed_df.reset_index(drop=True)

    return removed_df


def remove_bad_blocks(data_df: pd.DataFrame):
    """Function for removing blocks after condition rejection. Requirement: flux and one other condition still present.

    Args:
        data_df (pd.DataFrame): dataframe with conditions removed in preprocessing_utils.remove_bad_conditions

    Returns:
        pd.DataFrame: dataframe with blocks that do not meet the requirement of flux and one other condition present removed
    """

    # aggregate unique trial numbers in each block-condition group
    groupby_condition_df = (
        data_df[["Block", "Trial type", "Trial no"]]
        .groupby(["Block", "Trial type"])
        .agg("nunique")
    )
    groupby_condition_df.reset_index(inplace=True)

    # find blocks with no flux
    blocks_no_flux = [
        block
        for block in groupby_condition_df["Block"].unique()
        if "flux"
        not in groupby_condition_df["Trial type"][
            (groupby_condition_df["Block"] == block)
        ].to_list()
    ]

    # find blocks with one condition - this takes care of blocks where flux is the only one, shorter code than conditions on other-than-flux
    blocks_no_other = [
        block
        for block in groupby_condition_df["Block"].unique()
        if len(
            groupby_condition_df["Trial type"][(groupby_condition_df["Block"] == block)]
        )
        == 1
    ]

    # remove the identified bad blocks from dataframe
    removed_df = data_df[
        (~data_df["Block"].isin(blocks_no_flux))
        & (~data_df["Block"].isin(blocks_no_other))
    ]
    removed_df = removed_df.reset_index(drop=True)
    return removed_df


# Functions for removing artefacts

def remove_artefacts_non_physio_size(data_df: pd.DataFrame):
    """Function setting the non-physiological pupil size values in either eye to NaN.

    Args:
        data_df (pd.DataFrame): dataframe for one participant loaded using loading_utils.load_participant_data

    Returns:
        pd.DataFrame: dataframe with non-physiological pupil sizes set to pd.NA
    """
    data_df.loc[
        (data_df["Right - Size Mm"] < 1.5) | (data_df["Right - Size Mm"] > 9),
        "Right - Size Mm",
    ] = pd.NA
    data_df.loc[
        (data_df["Left - Size Mm"] < 1.5) | (data_df["Left - Size Mm"] > 9),
        "Left - Size Mm",
    ] = pd.NA
    data_df.loc[
        (data_df["Stim eye - Size Mm"] < 1.5) | (data_df["Stim eye - Size Mm"] > 9),
        "Stim eye - Size Mm",
    ] = pd.NA
    return data_df

def remove_artefacts_rolling_velocity_mad(
    resampled_df: pd.DataFrame,
    multiplier: float = 4.5,
    window: int = 60,
    column: str = "Stim eye - Size Mm",
):
    """Function for removing artefacts based on median absolute deviation threshold for absolute pupil velocity.
    Calculates the threshold in a rolling window.

    Args:
        resampled_df (pd.DataFrame): resampled dataframe from preprocessing_utils.resample_by_trial
        multiplier (float, optional): multiplier for MAD threshold (median+multiplier*MAD). Defaults to 4.5.
        window (int, optional): rolling window size in samples. Defaults to 60.
        column (str,optional): column with signal to remove artifacts from. Defaults to 'Stim eye - Size Mm'.

    Returns:
        pd.DataFrame: dataframe with pupil size in stimulated eye set to pd.NA where velocity exceeds threshold.
    """

    resampled_df["Time diff"] = resampled_df["Trial time Sec"].diff()
    resampled_df["Size diff"] = resampled_df[column].diff()
    resampled_df.loc[resampled_df["Time diff"] < 0, "Size diff"] = pd.NA
    resampled_df.loc[resampled_df["Time diff"] < 0, "Time diff"] = pd.NA

    for trial_no in sorted(resampled_df["Trial no"].unique()):
        trial = resampled_df[resampled_df["Trial no"] == trial_no].copy()
        trial["Pupil velocity -1"] = abs(trial["Size diff"] / trial["Time diff"])
        trial["Pupil velocity +1"] = abs(
            trial["Size diff"].shift(-1) / trial["Time diff"].shift(-1)
        )
        trial["Pupil velocity"] = trial[["Pupil velocity -1", "Pupil velocity +1"]].max(
            axis="columns"
        )

        median = (
            trial["Pupil velocity"]
            .rolling(window=window, min_periods=1, center=True)
            .median()
        )

        mad = (
            trial["Pupil velocity"]
            .rolling(window=window, min_periods=1, center=True)
            .apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
        )

        trial.loc[:, "MAD speed threshold"] = median + multiplier * mad
        resampled_df.loc[resampled_df['Trial no']==trial_no,'MAD speed threshold'] = trial['MAD speed threshold']
        resampled_df.loc[resampled_df['Trial no']==trial_no,'Pupil velocity'] = trial['Pupil velocity']
    
    resampled_df.loc[
        (
            (resampled_df['Pupil velocity'] > resampled_df["MAD speed threshold"])
        ),
        column,
    ] = pd.NA
    resampled_df = resampled_df.drop(columns=["Pupil velocity", "MAD speed threshold","Time diff","Size diff"])
    
    
    return resampled_df


def remove_artefacts_phase_velocity_mad(
    resampled_df: pd.DataFrame,
    multiplier: float = 4.5,
    column: str = "Stim eye - Size Mm",
):
    """Function for removing artefacts based on median absolute deviation threshold for absolute pupil velocity.
    Calculates a separate threshold for each phase of the trial: pre-stim, stim, post-stim.

    Args:
        resampled_df (pd.DataFrame): resampled dataframe from preprocessing_utils.resample_by_trial
        multiplier (float, optional): multiplier for MAD threshold (median+multiplier*MAD). Defaults to 4.5.
        column (str,optional): column with signal to remove artifacts from. Defaults to 'Stim eye - Size Mm'.

    Returns:
        pd.DataFrame: dataframe with pupil size in stimulated eye set to pd.NA where velocity exceeds threshold.
    """
    # calculate time difference and pupil size difference between samples, setting negative time differences to pd.NA (they indicate transition between trials)
    resampled_df["Time diff"] = resampled_df["Trial time Sec"].diff()
    resampled_df["Size diff"] = resampled_df[column].diff()
    resampled_df.loc[resampled_df["Time diff"] < 0, "Size diff"] = pd.NA
    resampled_df.loc[resampled_df["Time diff"] < 0, "Time diff"] = pd.NA

    # iterate by trials because the signal is not continuous outside of them (due to transition times etc.)
    for trial_no in sorted(resampled_df["Trial no"].unique()):
        
        trial = resampled_df[resampled_df["Trial no"] == trial_no].copy()
        
        # calculate maximum absolute pupil velocity for each sample (max[abs(v(t)),abs(v(t+1))])
        trial["Pupil velocity -1"] = abs(trial["Size diff"] / trial["Time diff"])
        trial["Pupil velocity +1"] = abs(
            trial["Size diff"].shift(-1) / trial["Time diff"].shift(-1)
        )
        trial["Pupil velocity"] = trial[["Pupil velocity -1", "Pupil velocity +1"]].max(
            axis="columns"
        )

        # iterate over trial phases to get MAD for each phase
        for phase in sorted(trial["Trial phase"].unique()):

            median = trial["Pupil velocity"][trial["Trial phase"] == phase].median()
            mad = (
                abs(trial["Pupil velocity"][trial["Trial phase"] == phase] - median)
            ).median()
            
            # get threshold and insert as a column in big dataframe
            threshold_up = median + multiplier * mad
            resampled_df.loc[
                (resampled_df["Trial no"] == trial_no)
                & (resampled_df["Trial phase"] == phase),
                "MAD speed threshold",
            ] = threshold_up
        
    
        resampled_df.loc[(resampled_df["Trial no"] == trial_no), "Pupil velocity"] = (
            trial["Pupil velocity"]
        )

    # replace samples above threshold with pd.NA
    resampled_df.loc[
        resampled_df["Pupil velocity"] > resampled_df["MAD speed threshold"], column
    ] = pd.NA
    
    # drop difference, velocity and threshold columns to retain only clean dataframe
    resampled_df = resampled_df.drop(columns=["Pupil velocity", "MAD speed threshold","Time diff","Size diff"])
    
    return resampled_df


def remove_artefacts_rolling_size_mad(
    resampled_df: pd.DataFrame,
    window: int = 60,
    multiplier: float = 4.5,
    column: str = "Stim eye - Size Mm",
):
    """Function for removing artefacts based on median absolute deviation threshold for pupil size. Calculates MAD in a rolling window.

    Args:
        resampled_df (pd.DataFrame): resampled dataframe from preprocessing_utils.resample_by_trial or with speed artefacts removed from remove_artifacts_phase_velocity_mad
        window (int): size of the rolling window in samples. Defaults to 60.
        multiplier (float, optional): multiplier for MAD threshold (median+/-multiplier*MAD). Defaults to 4.5.
        column (str, optional): Column with signal to remove artifacts from. Defaults to 'Stim eye - Size Mm'.

    Returns:
        pd.DataFrame: dataframe with pupil size samples out of MAD bounds replaced by pd.NA
    """
    # iterate by trials because the signal is not continuous outside of them (due to transition times etc.)
    for trial_no in sorted(resampled_df["Trial no"].unique()):
        trial = resampled_df[resampled_df["Trial no"] == trial_no].copy(deep=True)
        trial.reset_index(inplace=True)
        
        # calculate median in a rolling window
        median = (
            trial[column].rolling(window=window, min_periods=1, center=True).median()
        )

        # calculate mad in a rolling window
        mad = (
            trial[column]
            .rolling(window=window, min_periods=1, center=True)
            .apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
        )

        # get upper and lower threshold
        trial.loc[:, "MAD size upper threshold"] = median + multiplier * mad
        trial.loc[:, "MAD size lower threshold"] = median - multiplier * mad

        # put thresholds in big dataframe for this trial
        resampled_df.loc[
            resampled_df["Trial no"] == trial_no, "MAD size upper threshold"
        ] = trial["MAD size upper threshold"].to_list()
        resampled_df.loc[
            resampled_df["Trial no"] == trial_no, "MAD size lower threshold"
        ] = trial["MAD size lower threshold"].to_list()

    # replace values outside of threshold borders with pd.NA
    resampled_df.loc[
        (
            (resampled_df[column] > resampled_df["MAD size upper threshold"])
            | (resampled_df[column] < resampled_df["MAD size lower threshold"])
        ),
        column,
    ] = pd.NA

    # drop threshold columns to retain a clean dataframe
    resampled_df = resampled_df.drop(
        columns=["MAD size upper threshold", "MAD size lower threshold"]
    )

    return resampled_df

# Others (WIP)

def calculate_change_from_baseline(data_df):
    data_df["Baseline change %"] = pd.Series()
    for i in data_df["Trial no"].unique():
        trial_df = data_df[(data_df["Trial no"] == i)].copy()
        baseline = trial_df["Stim eye - Size Mm"][trial_df["Trial time Sec"] < 0].mean()
        data_df.loc[(data_df["Trial no"] == i), "Baseline change %"] = (
            (trial_df["Stim eye - Size Mm"] - baseline) * 100 / baseline
        )
    return data_df


class Preprocessor(ABC):
    def __init__(self, data_df):
        self.data_df = data_df.copy()
