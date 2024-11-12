import pandas as pd
import numpy as np
import datetime


def resample_by_trial(data_df, sample_freq=50):
    # get time step in ms from sampling frequency provided
    time_step = np.ceil((1000 / sample_freq) * 1e6)

    # take subset of data without transition and adaptation parts
    data_subset = data_df[
        (data_df["Trial phase"] != "Adaptation")
        & (data_df["Trial phase"] != "Transition")
    ]

    # map trial-relevant variables to trial numbers for trial marking after resampling
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

    # make datetime index for resampling
    data_subset["Trial time datetime"] = data_subset["Trial time Sec"].apply(
        lambda x: datetime.timedelta(seconds=x)
    )
    data_subset.set_index("Trial time datetime", inplace=True)

    # remove rows with NaN in stimulated eye pupil size
    data_subset = data_subset[data_subset["Stim eye - Size Mm"].notna()]

    # resample by trial and create a new dataframe
    trials_for_new_df = []
    for i, trial_no in enumerate(trial_list):

        trial = data_subset[["Trial time Sec", "Stim eye - Size Mm"]][
            data_subset["Trial no"] == trial_no
        ].copy()
        trial.loc[datetime.timedelta(seconds=-1)] = (
            pd.Series()
        )  # add a row at -1s so that every trial has the same time ticks
        trial.loc[datetime.timedelta(seconds=18)] = (
            pd.Series()
        )  # just in case the trial is too short, add row at 18s
        resampled_trial = trial.resample(str(time_step) + "ns").agg(
            {"Stim eye - Size Mm": "mean"}
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
        resampled_trial["Trial phase"] = ["N/A"] * len(resampled_trial)
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


def remove_trials_below_percentage(
    resampled_df,
    baseline_threshold=40,
    poi_threshold=75,
    baseline_time=[-1, 0],
    poi_time=[0, 6],
):

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
    thresholded_df, fs=30, max_nan_length=500, poi_time=[0, 6]
):
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


def remove_bad_conditions(data_df, trial_min=3):
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

    return removed_df


def remove_bad_blocks(data_df, trial_min=3):

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
    return removed_df


def calculate_change_from_baseline(data_df):
    data_df["Baseline change %"] = pd.Series()
    for i in data_df["Trial no"].unique():
        trial_df = data_df[(data_df["Trial no"] == i)].copy()
        baseline = trial_df["Stim eye - Size Mm"][trial_df["Trial time Sec"] < 0].mean()
        data_df.loc[(data_df["Trial no"] == i), "Baseline change %"] = (
            (trial_df["Stim eye - Size Mm"] - baseline) * 100 / baseline
        )
    return data_df
