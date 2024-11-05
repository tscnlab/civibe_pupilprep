import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def mark_not_measured(data_df):
    data_df['Stim eye - Measured'] = [False]*len(data_df)
    data_df.loc[ data_df['Stim eye - Size Mm'].notna(),'Stim eye - Measured'] = True
    data_df.loc[(data_df['Right - Size Mm'].isna())&(data_df['Left - Size Mm'].isna()),'Stim eye - Measured'] = 'missing'
    return data_df

def resample_by_trial(data_df,sample_freq = 50):
    # get time step in ms from sampling frequency provided
    time_step = 1000/sample_freq
    
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

    # resample by trial and create a new dataframe
    trials_for_new_df = []
    for i, trial_no in enumerate(trial_list):

        trial = data_subset[["Trial time Sec", "Stim eye - Size Mm"]][
            data_subset["Trial no"] == trial_no
        ].copy()
        trial.loc[datetime.timedelta(seconds=-1)] = (
            pd.Series()
        )  # add a row at -1s so that every trial has the same time ticks

        resampled_trial = trial.resample(str(time_step)+'ms').agg({"Stim eye - Size Mm": "mean"})
        # cut trial to 18 s
        resampled_trial=resampled_trial[datetime.timedelta(seconds=-1):datetime.timedelta(seconds=18)]
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


def calculate_change_from_baseline(data_df):
    data_df["Baseline change %"] = [pd.NA] * len(data_df)
    for i in data_df["Trial no"][data_df["Trial no"].notna()].unique():
        trial_df = data_df[(data_df["Trial no"] == i)].copy()
        baseline = trial_df["Stim eye - Size Mm"][trial_df["Trial time Sec"] < 0].mean()
        data_df.loc[(data_df["Trial no"] == i), "Baseline change %"] = (
            (trial_df["Stim eye - Size Mm"] - baseline) * 100 / baseline
        )
    return data_df


def plot_trials(data_df, participant_id, trial_types: list):
    fig, axs = plt.subplots(1, len(trial_types), figsize=(len(trial_types) * 5, 5))
    for i, trial_type in enumerate(trial_types):
        axs[i].axvspan(0, 5, alpha=0.1, color="green")
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
        axs[i].axvspan(0, 5, alpha=0.1, color="green")
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
