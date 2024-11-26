import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Visualisations of artifact removal functions


def plot_velocity_MAD(
    resampled_df: pd.DataFrame, trials_to_vis: list, multiplier: float = 4.5
):
    """Function that plots for selected trials: pupil size, pupil velocity (absolute), MAD threshold for pupil velocity for outlier detection. Returns nothing.

    Args:
        resampled_df (pd.DataFrame): resampled dataframe from preprocessing_utils.resample_by_trial
        trials_to_vis (list): list of trial numbers to visualize (from Trial no column)
        multiplier (float, optional): multiplier for MAD threshold (threshold=median+multiplier*MAD). Defaults to 4.5.
    """

    resampled_df["Time diff"] = resampled_df["Trial time Sec"].diff()
    resampled_df["Size diff"] = resampled_df["Stim eye - Size Mm"].diff()
    resampled_df.loc[resampled_df["Time diff"] < 0, "Size diff"] = pd.NA
    resampled_df.loc[resampled_df["Time diff"] < 0, "Time diff"] = pd.NA

    for trial_no in trials_to_vis:
        trial = resampled_df[resampled_df["Trial no"] == trial_no].copy()
        trial["Pupil velocity -1"] = abs(trial["Size diff"] / trial["Time diff"])
        trial["Pupil velocity +1"] = abs(
            trial["Size diff"].shift(-1) / trial["Time diff"].shift(-1)
        )
        trial["Pupil velocity"] = trial[["Pupil velocity -1", "Pupil velocity +1"]].max(
            axis="columns"
        )

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

        plt.figure(figsize=(30, 10))

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
            label="mad threshold speed mm",
        )

        plt.ylim([0, 10])
        plt.grid(which="both")
        plt.minorticks_on()
        plt.title(str(trial_no))
        plt.legend()
        plt.xlabel("Time [s]")
        plt.show()


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

    for trial_no in trials_to_vis:
        trial = resampled_df[resampled_df["Trial no"] == trial_no].copy(deep=True)
        trial.reset_index(inplace=True)
        trial["MAD size threshold"] = pd.Series()

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


# Functions for plotting trials

def plot_trials(data_df, participant_id, trial_types: list):
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
