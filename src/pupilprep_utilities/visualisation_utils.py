import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import pupilprep_utilities.preprocessing_utils as prep

# Visualisations of artifact removal functions


def plot_phase_velocity_MAD(
    resampled_df: pd.DataFrame, trials_to_vis: list, multiplier: float = 6
):
    """Function that plots for selected trials: pupil size from stimulated eye, pupil velocity (absolute), MAD threshold for pupil velocity for outlier detection. Returns nothing.

    Args:
        resampled_df (pd.DataFrame): resampled dataframe from preprocessing_utils.resample_by_trial
        trials_to_vis (list): list of trial numbers to visualize (from Trial no column)
        multiplier (float, optional): multiplier for MAD threshold (threshold=median+multiplier*MAD). Defaults to 6.
    """
    # get time and size differences between samples
    resampled_df["Time diff"] = resampled_df["Trial time Sec"].diff()
    resampled_df["Size diff"] = resampled_df["Stim eye - Size Mm"].diff()
    resampled_df.loc[resampled_df["Time diff"] < 0, "Size diff"] = pd.NA
    resampled_df.loc[resampled_df["Time diff"] < 0, "Time diff"] = pd.NA

    # iterate over trials to visualize
    for trial_no in trials_to_vis:
        trial = resampled_df[resampled_df["Trial no"] == trial_no].copy()
        # calculate max velocity for a sample max[abs(v(t)),abs(v(t+1))]
        trial["Pupil velocity -1"] = abs(trial["Size diff"] / trial["Time diff"])
        trial["Pupil velocity +1"] = abs(
            trial["Size diff"].shift(-1) / trial["Time diff"].shift(-1)
        )
        trial["Pupil velocity"] = trial[["Pupil velocity -1", "Pupil velocity +1"]].max(
            axis="columns"
        )

        # iterate over trial phases and calculate mad threshold for each phase
        for phase in sorted(trial["Trial phase"].unique()):

            median = trial["Pupil velocity"][trial["Trial phase"] == phase].median()
            mad = (
                abs(trial["Pupil velocity"][trial["Trial phase"] == phase] - median)
            ).median()
            threshold_up = median + multiplier * mad

            resampled_df.loc[
                (resampled_df["Trial no"] == trial_no)
                & (resampled_df["Trial phase"] == phase),
                "MAD speed threshold",
            ] = threshold_up
        resampled_df.loc[(resampled_df["Trial no"] == trial_no), "Pupil velocity"] = (
            trial["Pupil velocity"]
        )

        # plot signal, velocity, threshold
        plt.figure(figsize=(20, 7))

        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["Pupil velocity"][resampled_df["Trial no"] == trial_no],
            label="speed mm/s",
            marker=".",
            linestyle="none",
        )
        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["Stim eye - Size Mm"][resampled_df["Trial no"] == trial_no],
            label="size mm",
            marker=".",
            linestyle="none",
        )
        plt.plot(
            resampled_df["Trial time Sec"][
                (resampled_df["Trial no"] == trial_no)
                & (resampled_df["Pupil velocity"] > resampled_df["MAD speed threshold"])
            ],
            resampled_df["Stim eye - Size Mm"][
                (resampled_df["Trial no"] == trial_no)
                & (resampled_df["Pupil velocity"] > resampled_df["MAD speed threshold"])
            ],
            marker=".",
            linestyle="none",
            color="k",
            label="Removed by MAD",
        )
        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["MAD speed threshold"][resampled_df["Trial no"] == trial_no],
            label="mad threshold speed mm/s",
        )

        plt.ylim([0, 10])
        plt.grid(which="both")
        plt.minorticks_on()
        plt.title(str(trial_no))
        plt.legend()
        plt.xlabel("Time [s]")
        plt.show()
    # drop velocity and threshold columns to retain only clean dataframe
    resampled_df = resampled_df.drop(
        columns=["Pupil velocity", "MAD speed threshold", "Time diff", "Size diff"]
    )


def plot_rolling_velocity_MAD(
    resampled_df: pd.DataFrame, trials_to_vis: list, window=60, multiplier: float = 6
):
    """Function that plots for selected trials: pupil size from stimulated eye, pupil velocity (absolute), MAD threshold for pupil velocity for outlier detection. Returns nothing.

    Args:
        resampled_df (pd.DataFrame): resampled dataframe from preprocessing_utils.resample_by_trial
        trials_to_vis (list): list of trial numbers to visualize (from Trial no column)
        window (int, optional): window size for rolling MAD calculation in samples. Defaults to 60.
        multiplier (float, optional): multiplier for MAD threshold (threshold=median+multiplier*MAD). Defaults to 6.
    """
    # get time and size differences between samples
    resampled_df["Time diff"] = resampled_df["Trial time Sec"].diff()
    resampled_df["Size diff"] = resampled_df["Stim eye - Size Mm"].diff()
    resampled_df.loc[resampled_df["Time diff"] < 0, "Size diff"] = pd.NA
    resampled_df.loc[resampled_df["Time diff"] < 0, "Time diff"] = pd.NA

    # iterate over trials to visualize
    for trial_no in trials_to_vis:
        trial = resampled_df[resampled_df["Trial no"] == trial_no].copy()
        # calculate max velocity for a sample max[abs(v(t)),abs(v(t+1))]
        trial["Pupil velocity -1"] = abs(trial["Size diff"] / trial["Time diff"])
        trial["Pupil velocity +1"] = abs(
            trial["Size diff"].shift(-1) / trial["Time diff"].shift(-1)
        )
        trial["Pupil velocity"] = trial[["Pupil velocity -1", "Pupil velocity +1"]].max(
            axis="columns"
        )
        # get rolling MAD
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
        resampled_df.loc[
            resampled_df["Trial no"] == trial_no, "MAD speed threshold"
        ] = trial["MAD speed threshold"]
        resampled_df.loc[resampled_df["Trial no"] == trial_no, "Pupil velocity"] = (
            trial["Pupil velocity"]
        )

        # plot signal, velocity, threshold
        plt.figure(figsize=(20, 7))

        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["Pupil velocity"][resampled_df["Trial no"] == trial_no],
            label="speed mm/s",
            marker=".",
            linestyle="none",
        )
        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["Stim eye - Size Mm"][resampled_df["Trial no"] == trial_no],
            label="size mm",
            marker=".",
            linestyle="none",
        )
        plt.plot(
            resampled_df["Trial time Sec"][
                (resampled_df["Trial no"] == trial_no)
                & (resampled_df["Pupil velocity"] > resampled_df["MAD speed threshold"])
            ],
            resampled_df["Stim eye - Size Mm"][
                (resampled_df["Trial no"] == trial_no)
                & (resampled_df["Pupil velocity"] > resampled_df["MAD speed threshold"])
            ],
            marker=".",
            linestyle="none",
            color="k",
            label="Removed by MAD",
        )
        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["MAD speed threshold"][resampled_df["Trial no"] == trial_no],
            label="mad threshold speed mm/s",
        )

        plt.ylim([0, 10])
        plt.grid(which="both")
        plt.minorticks_on()
        plt.title(str(trial_no))
        plt.legend()
        plt.xlabel("Time [s]")
        plt.show()
    # drop velocity and threshold columns to retain only clean dataframe
    resampled_df = resampled_df.drop(
        columns=["Pupil velocity", "MAD speed threshold", "Time diff", "Size diff"]
    )


def plot_rolling_size_MAD(
    resampled_df: pd.DataFrame,
    trials_to_vis: list,
    window: int = 60,
    multiplier: float = 4.5,
):
    """Function for plotting pupil size MAD thresholds for selected trials. Returns nothing.

    Args:
        resampled_df (pd.DataFrame): resampled dataframe from preprocessing_utils.resample_by_trial, or with velocity artifacts removed by preprocessing_utils.remove_artifacts_phase_velocity_mad
        trials_to_vis (list): list of trial numbers to visualize (based on Trial no column)
        window (int, optional): window size for rolling MAD calculation in samples. Defaults to 60.
        multiplier (float, optional): multiplier for MAD threshold (threshold=median+/-multiplier*MAD). Defaults to 4.5.
    """
    # iterate over trials to visualize
    for trial_no in trials_to_vis:
        trial = resampled_df[resampled_df["Trial no"] == trial_no].copy(deep=True)
        trial.reset_index(inplace=True)

        # calculate mad threshold, upper and lower in rolling window
        median = (
            trial["Stim eye - Size Mm"]
            .rolling(window=window, min_periods=1, center=True)
            .median()
        )

        mad = (
            trial["Stim eye - Size Mm"]
            .rolling(window=window, min_periods=1, center=True)
            .apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
        )

        trial.loc[:, "MAD size upper threshold"] = median + multiplier * mad
        trial.loc[:, "MAD size lower threshold"] = median - multiplier * mad

        resampled_df.loc[
            resampled_df["Trial no"] == trial_no, "MAD size upper threshold"
        ] = trial["MAD size upper threshold"].to_list()
        resampled_df.loc[
            resampled_df["Trial no"] == trial_no, "MAD size lower threshold"
        ] = trial["MAD size lower threshold"].to_list()

        # plot signal, thresholds, samples removed by threshold
        plt.figure(figsize=(30, 10))

        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["Stim eye - Size Mm"][resampled_df["Trial no"] == trial_no],
            label="size mm",
            marker=".",
            linestyle="none",
        )
        plt.plot(
            resampled_df["Trial time Sec"][
                (resampled_df["Trial no"] == trial_no)
                & (
                    resampled_df["Stim eye - Size Mm"]
                    > resampled_df["MAD size upper threshold"]
                )
            ],
            resampled_df["Stim eye - Size Mm"][
                (resampled_df["Trial no"] == trial_no)
                & (
                    resampled_df["Stim eye - Size Mm"]
                    > resampled_df["MAD size upper threshold"]
                )
            ],
            marker=".",
            linestyle="none",
            color="k",
            label="Removed by MAD",
        )
        plt.plot(
            resampled_df["Trial time Sec"][
                (resampled_df["Trial no"] == trial_no)
                & (
                    resampled_df["Stim eye - Size Mm"]
                    < resampled_df["MAD size lower threshold"]
                )
            ],
            resampled_df["Stim eye - Size Mm"][
                (resampled_df["Trial no"] == trial_no)
                & (
                    resampled_df["Stim eye - Size Mm"]
                    < resampled_df["MAD size lower threshold"]
                )
            ],
            marker=".",
            linestyle="none",
            color="k",
            label="Removed by MAD",
        )
        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["MAD size upper threshold"][
                resampled_df["Trial no"] == trial_no
            ],
            label="mad threshold size up mm",
        )
        plt.plot(
            resampled_df["Trial time Sec"][resampled_df["Trial no"] == trial_no],
            resampled_df["MAD size lower threshold"][
                resampled_df["Trial no"] == trial_no
            ],
            label="mad threshold size low mm",
        )

        plt.ylim([0, 10])
        plt.grid(which="both")
        plt.minorticks_on()
        plt.legend()
        plt.title(str(trial_no))
        plt.xlabel("Time [s]")
        plt.show()

    # drop unnecessary columns
    resampled_df = resampled_df.drop(
        columns=["MAD size upper threshold", "MAD size lower threshold"]
    )


# Functions for EDA visualization of trials and blocks remaining after trial rejection based on percentage criteria and NaN criteria


def plot_grid_stim_vs_baseline(
    data_dir: str,
    data_suffix: str,
    participant_list: list,
    threshold_step: float,
    poi_time: list = [0, 6],
    baseline_time: list = [-1, 0],
):
    """Function for plotting grid of joint trial acceptance probability depending on threshold value for period of interest and baseline.

    Args:
        data_dir (str): directory with resampled data from all participants
        data_suffix (str): suffix of data filenames (e.g. _30_resampled_data.csv)
        participant_list (list): list of participant numbers to calculate the thresholds for
        threshold_step (float): by what value to stagger the thresholds in the grid (between 0 and 1, e.g. 0.1 = 10%)
        poi_time (list): [start, end] of period of interest in seconds. Defaults to [0,6].
        baseline_time (list): [start,end] of baseline period in seconds. Defaults to [-1,0].

    Returns:
        dict: dictionary of grids for participants
        fig, axs - Matplotlib figure and axes objects
    """
    baseline_stim_grid_arrays = {}

    for participant_id in participant_list:
        data_path = os.path.join(data_dir, str(participant_id) + data_suffix)
        data_df_new = pd.read_csv(data_path)

        baseline_df_new = data_df_new[data_df_new["Trial phase"] == "pre-stim"].copy()
        data_df_new = data_df_new[
            (data_df_new["Trial time Sec"] >= poi_time[0])
            & (data_df_new["Trial time Sec"] <= poi_time[1])
        ].copy()

        groupby_df_stim = (
            data_df_new[["Block", "Trial no", "Trial type", "Stim eye - Size Mm"]]
            .groupby(["Block", "Trial no", "Trial type"])
            .agg(["count", "size"])
            .reset_index()
        )
        groupby_df_stim[("Stim eye - Size Mm", "count/size ratio")] = (
            groupby_df_stim[("Stim eye - Size Mm", "count")]
            / groupby_df_stim[("Stim eye - Size Mm", "size")]
        )

        groupby_df_baseline = (
            baseline_df_new[["Block", "Trial no", "Trial type", "Stim eye - Size Mm"]]
            .groupby(["Block", "Trial no", "Trial type"])
            .agg(["count", "size"])
            .reset_index()
        )
        groupby_df_baseline[("Stim eye - Size Mm", "count/size ratio")] = (
            groupby_df_baseline[("Stim eye - Size Mm", "count")]
            / groupby_df_baseline[("Stim eye - Size Mm", "size")]
        )

        threshold_range = np.arange(0, 1 + threshold_step, threshold_step)
        baseline_stim_grid = np.zeros(
            [
                len(data_df_new["Block"].unique()),
                int(1 / threshold_step) + 1,
                int(1 / threshold_step) + 1,
            ]
        )
        for k, block in enumerate(sorted(groupby_df_stim["Block", ""].unique())):
            trial_count = len(
                groupby_df_stim[("Trial no", "")][
                    (groupby_df_stim[("Block", "")] == block)
                ].unique()
            )

            for i, threshold in enumerate(threshold_range):
                stim_above_threshold = (
                    groupby_df_stim[("Stim eye - Size Mm", "count/size ratio")][
                        (groupby_df_stim[("Block", "")] == block)
                    ]
                    >= threshold
                )
                for j, threshold in enumerate(threshold_range):
                    baselines_above_threshold = (
                        groupby_df_baseline[("Stim eye - Size Mm", "count/size ratio")][
                            (groupby_df_baseline[("Block", "")] == block)
                        ]
                        >= threshold
                    )
                    trials_accepted_both = (
                        stim_above_threshold & baselines_above_threshold
                    ).sum()
                    baseline_stim_grid[k, i, j] = (
                        trials_accepted_both / trial_count
                    ) * 100  # percentage of trials accepted based on both baseline and POI threshold

        baseline_stim_grid_arrays[str(participant_id)] = baseline_stim_grid

    fig, axs = plt.subplots(
        len(participant_list),
        len(data_df_new["Block"].unique()),
        figsize=(35, 35),
        sharex=True,
        sharey=True,
    )
    for i, participant_id in enumerate(participant_list):
        participant_array = baseline_stim_grid_arrays[str(participant_id)]
        for j, block in enumerate(np.arange(0, len(data_df_new["Block"].unique()))):
            im = axs[i, j].imshow(
                participant_array[j],
                extent=[0, 100, 0, 100],
                origin="lower",
                vmin=0,
                vmax=100,
            )

            if j == 10:
                plt.colorbar(im, ax=axs[i, j])
            if i == 0:
                axs[i, j].set_title("Block " + str(block))
            if j == 0:
                axs[i, j].set_ylabel(str(participant_id) + " - POI threshold %")
            if i == 11:
                axs[i, j].set_xlabel("baseline threshold %")

    plt.tight_layout()
    plt.show()
    return baseline_stim_grid_arrays, fig, axs


def plot_trials_remaining_at_nan_threshold(
    data_dir: str,
    data_suffix: str,
    participant_list: list,
    fs: int = 30,
    nan_threshold_step: int = 50,
    baseline_threshold: int = 40,
    poi_threshold: int = 75,
    baseline_time: list = [-1, 0],
    poi_time: list = [0, 6],
):
    """Function for plotting trajectories of trials remaining for conditions depending on NaN sequence length threshold.

    Args:
        data_dir (str): directory with resampled/cleaned data
        data_suffix (str): suffix for data to be analyzed, e.g. "_30_resampled_data.csv"
        participant_list (list): list of participants
        fs (int, optional): Resampling frequency. Defaults to 30.
        nan_threshold_step (int, optional): NaN threshold step in ms. Defaults to 50.
        baseline_threshold (int, optional): Percentage completeness threshold for baseline. Set to 0 if you want to plot without % thresholding. Defaults to 40.
        poi_threshold (int, optional): Percentage completeness threshold for period of interest. Set to 0 if you want to plot without % thresholding. Defaults to 75.
        baseline_time (list, optional): [start,end] of baseline period in seconds. Defaults to [-1, 0].
        poi_time (list, optional): [start,end] of period of interest in seconds. Defaults to [0, 6].

    Returns:
        dict: dictionary with dataframes with trials remaining after NaN thresholding
        fig,axs: Matplotlib figure and axes objects with the plot
    """
    acceptance_threshold_dfs = {}

    for participant_id in participant_list:
        data_path = os.path.join(data_dir, str(participant_id) + data_suffix)
        data_df = pd.read_csv(data_path)

        thresholded_df = prep.remove_trials_below_percentage(
            data_df, baseline_threshold, poi_threshold, baseline_time, poi_time
        )

        blocks = []
        trial_types = []
        thresholds = []
        trials_remaining_list = []

        threshold_range = np.arange(
            50, 2000 + nan_threshold_step, nan_threshold_step
        )  # limit thresholds to analyze to 2 seconds to save on computation time
        for threshold in threshold_range:
            no_nan_df = prep.remove_trials_with_long_nans(
                thresholded_df, fs=fs, max_nan_length=threshold, poi_time=poi_time
            )
            groupby_df = (
                no_nan_df[["Block", "Trial type", "Trial no"]]
                .groupby(["Block", "Trial type"])
                .agg(["nunique"])
            )
            groupby_df.reset_index(inplace=True)
            trials_remaining_list.append(groupby_df[("Trial no", "nunique")].to_list())
            thresholds.append([threshold] * len(groupby_df))
            blocks.append(groupby_df[("Block", "")].to_list())
            trial_types.append(groupby_df[("Trial type", "")].to_list())

        thresholds = sum(thresholds, [])
        blocks = sum(blocks, [])
        trials_remaining_list = sum(trials_remaining_list, [])
        trial_types = sum(trial_types, [])
        acceptance_threshold_df = pd.DataFrame(
            {
                "Block": blocks,
                "Trial type": trial_types,
                "Threshold [s]": thresholds,
                "Trials above threshold": trials_remaining_list,
            }
        )
        acceptance_threshold_df["Threshold [s]"] = acceptance_threshold_df[
            "Threshold [s]"
        ].apply(lambda x: x / 1000)
        acceptance_threshold_df.loc[
            acceptance_threshold_df["Trials above threshold"].isna(),
            "Trials above threshold",
        ] = 0

        acceptance_threshold_dfs[str(participant_id)] = acceptance_threshold_df

    fig, axs = plt.subplots(
        len(participant_list),
        len(data_df["Block"].unique()),
        figsize=(35, 35),
        sharex=True,
        sharey=True,
    )
    for i, participant_id in enumerate(participant_list):
        participant_df = acceptance_threshold_dfs[str(participant_id)]
        for j, block in enumerate(len(data_df["Block"].unique())):
            try:
                block_df = participant_df[participant_df["Block"] == block]
                axs[i, j].fill_between(
                    x=np.arange(
                        participant_df["Threshold [s]"].min(),
                        participant_df["Threshold [s]"].max()
                        + nan_threshold_step / 1000,
                        nan_threshold_step / 1000,
                    ),
                    y1=3,
                    color="k",
                    alpha=0.1,
                    label="<3 trials",
                )
                axs[i, j].axvline(x=0.5, label="0.5 s", color="k", ls="--")
                sns.lineplot(
                    data=block_df,
                    x="Threshold [s]",
                    y="Trials above threshold",
                    ax=axs[i, j],
                    hue="Trial type",
                    estimator=None,
                    n_boot=0,
                )
            except ValueError:
                pass  # pass instead of continue so that the block number and label get printed even if there is nothing to plot

            if i != 0 or j != 0:
                axs[i, j].legend([])
            if i == 0:
                axs[i, j].set_title("Block " + str(block))
            if j == 0:
                axs[i, j].set_ylabel(str(participant_id) + " - no of trials accepted")

    plt.tight_layout()
    plt.show()
    return acceptance_threshold_dfs, fig, axs


def plot_condition_availability(
    completeness_df: pd.DataFrame,
    participant_list: list = [
        200,
        201,
        202,
        204,
        205,
        206,
        207,
        209,
        210,
        211,
        212,
        213,
    ],
    conditions: list = ["flux", "l-m", "lms", "mel", "s"],
):
    """Function that plots condition availability in blocks.

    Args:
        completeness_df (pd.DataFrame): dataframe created with analysis_utils.make_completeness_stats_df
        participant_list (list, optional): List of participants to plot. Defaults to [200, 201, 202, 204, 205, 206, 207, 209, 210, 211, 212, 213].
        conditions (list, optional): List of conditions. Defaults to ["flux", "l-m", "lms", "mel", "s"].

    Returns:
        fig,axes: Matplotlib figure and axes objects
    """

    fig, axes = plt.subplots(
        2, np.ceil(participant_list / 2), figsize=(15, 5), sharex=True, sharey=True
    )
    for participant_id, ax in zip(participant_list, axes.flat):
        if participant_id in completeness_df["Participant"].unique():
            subset_df = completeness_df[
                completeness_df["Participant"] == participant_id
            ]
            subset_df.reset_index(inplace=True)
            subset_df.loc[subset_df["Trial count"] == "less than 3", "Trial count"] = (
                pd.NA
            )
            for i, cond in enumerate(sorted(subset_df["Condition"].unique())):
                subset_df.loc[
                    (subset_df["Condition"] == cond)
                    & (subset_df["Trial count"].notna()),
                    "Trial count",
                ] = i

            sns.pointplot(
                ax=ax,
                data=subset_df,
                x="Block",
                y="Trial count",
                hue="Condition",
                linewidth=1.5,
                markersize=2.3,
            )
        ax.set_title("Participant " + str(participant_id))
        ax.set_ylabel("Condition")
        ax.set_yticks(np.arange(len(conditions)), labels=conditions)
        ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()
    return fig, axes


# Functions for plotting trials


def plot_trials(data_df: pd.DataFrame, participant_id: int, trial_types: list):
    """Function for plotting trials.

    Args:
        data_df (pd.DataFrame): dataframe with pupil size data, has columns 'Stim eye - Size Mm' and 'Trial type'
        participant_id (int): participant number to plot
        trial_types (list): list of trial types to plot

    Returns:
        fig,axs: figure and axes of the plot
    """
    fig, axs = plt.subplots(1, len(trial_types), figsize=(len(trial_types) * 5, 5))
    for i, trial_type in enumerate(trial_types):
        axs[i].axvspan(0, 5, alpha=0.1, color="green", label="Stim phase")
        sns.scatterplot(
            ax=axs[i],
            data=data_df[data_df["Trial type"] == trial_type],
            x="Trial time Sec",
            y="Stim eye - Size Mm",
            hue="Trial no",
            s=0.5,
        )
        axs[i].set_title(trial_type)
        if i > 0:
            axs[i].set_ylabel("")
        axs[i].legend([], [], frameon=False)
        axs[i].set_ylim(
            [
                0,
                np.max(
                    data_df["Stim eye - Size Mm"][data_df["Trial type"] == trial_type]
                )
                + 1,
            ]
        )
    fig.suptitle(f"Pupil size per trial for participant {participant_id}")
    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_baseline_change(data_df, participant_id, trial_types: list):
    fig, axs = plt.subplots(1, len(trial_types), figsize=(len(trial_types) * 5, 5))
    for i, trial_type in enumerate(trial_types):
        axs[i].axvspan(0, 5, alpha=0.1, color="green", label="Stim phase")
        sns.scatterplot(
            ax=axs[i],
            data=data_df[data_df["Trial type"] == trial_type],
            x="Trial time Sec",
            y="Baseline change %",
            hue="Trial no",
            s=0.5,
        )
        axs[i].set_title(trial_type)
        if i > 0:
            axs[i].set_ylabel("")
        axs[i].legend([], [], frameon=False)
        axs[i].set_ylim(
            [
                np.min(
                    data_df["Baseline change %"][data_df["Trial type"] == trial_type]
                )
                - 1,
                np.max(
                    data_df["Baseline change %"][data_df["Trial type"] == trial_type]
                )
                + 1,
            ]
        )
    fig.suptitle(
        f"Change in stimulated pupil size from baseline average per trial for participant {participant_id}"
    )
    plt.tight_layout()
    plt.show()
    return fig, axs
