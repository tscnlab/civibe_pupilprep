import pandas as pd
import numpy as np
import seaborn as sns
import os


def make_filepaths(rootdir: str):
    """
    Function for extracting paths to protocol and data files
    rootdir - string, path to directory with subject experiment files
    Returns:
    fp_protocol - string, path to protocol csv file
    fp_recording - list of strings, paths to separate adaptation/trial recording csv files
    fp_whole_exp - string, path to the whole test csv recording
    """
    fp_protocol = [
        os.path.join(subdirs, file)
        for subdirs, dirs, files in os.walk(rootdir)
        for file in files
        if (file.endswith(".csv")) & ("sine" in file)
    ][0]

    fp_recording = [
        os.path.join(subdirs, file)
        for subdirs, dirs, files in os.walk(rootdir)
        for file in files
        if (file.endswith("data.csv")) & ("Sequence_" in subdirs)
    ]
    fp_whole_exp = [
        os.path.join(subdirs, file)
        for subdirs, dirs, files in os.walk(rootdir)
        for file in files
        if (file.endswith("data.csv")) & ("Sequence_" not in subdirs)
    ][0]

    return fp_protocol, fp_recording, fp_whole_exp


def mark_phases(data_df, fp_protocol):
    """
    Function for marking phases in the experiment.
    Input:
    data_df - dataframe with experiment data from one recording of one participant
    fp_protocol - filepath to protocol file for the recording
    Returns:
    data_df - dataframe from input with an added column 'Phase' with positions: Adaptation, pre-stim, stim, post-stim and Transition
    """
    data_df["Phase"] = ["N/A"] * len(data_df)

    for seq_id in data_df["Sequence index"].unique():

        seq_length = data_df["Sequence time Sec"][
            data_df["Sequence index"] == seq_id
        ].max()

        data_df.loc[
            (data_df["Sequence index"] == seq_id)
            & (data_df["Sequence time Sec"] >= seq_length - 1),
            "Phase",
        ] = "pre-stim"

        if seq_id == 1:
            data_df.loc[
                (data_df["Sequence index"] == seq_id) & (data_df["Phase"] == "N/A"),
                "Phase",
            ] = "Adaptation"
        else:
            if "left" in fp_protocol:
                data_df.loc[
                    (data_df["Sequence index"] == seq_id)
                    & (data_df["Excitation label - Left"] != "baseline"),
                    "Phase",
                ] = "stim"
            else:
                data_df.loc[
                    (data_df["Sequence index"] == seq_id)
                    & (data_df["Excitation label - Right"] != "baseline"),
                    "Phase",
                ] = "stim"
            data_df.loc[
                (data_df["Sequence index"] == seq_id)
                & (data_df["Experiment state"] == "Passive"),
                "Phase",
            ] = "Transition"
            data_df.loc[data_df["Phase"] == "N/A", "Phase"] = "post-stim"
    return data_df


def make_protocol_dfs(fp_protocol: str):
    """
    Function for making protocol information dataframes
    fp_protocol - string, path to csv file with protocol data
    Returns:
    protocol_vars_df - DataFrame, includes the first lines with static protocol information plus information on dominant eye
    protocol_timecourse_df - DataFrame, includes the timecourse of the protocol data, with added 'Eye' column - dominant eye L or R
    """
    protocol_vars_df = pd.read_csv(
        fp_protocol,
        skiprows=np.arange(12, 10e6),
        delimiter=";",
        names=["Var", "Val 1", "Val 2"],
        usecols=[i for i in range(3)],
    )

    protocol_timecourse_df = pd.read_csv(
        fp_protocol, skiprows=np.arange(0, 12), delimiter=";"
    )
    if "left" in fp_protocol:
        protocol_timecourse_df["Eye"] = [
            "L" for i in range(len(protocol_timecourse_df))
        ]
        protocol_vars_df.loc[len(protocol_vars_df)] = ["Dominant eye", "L", np.nan]
    else:
        protocol_timecourse_df["Eye"] = [
            "R" for i in range(len(protocol_timecourse_df))
        ]
        protocol_vars_df.loc[len(protocol_vars_df)] = ["Dominant eye", "R", np.nan]

    return protocol_vars_df, protocol_timecourse_df


def make_whole_exp_df(fp_whole_exp: str, fp_protocol: str):
    """
    Function for making a dataframe from the whole test recording
    fp_whole_exp - string, path to csv file with whole test data
    fp_protocol - string, path to csv with protocol data (for eye information)
    Returns:
    data_df - DataFrame with data from the whole test recording, with an added column 'Eye' - dominant eye L or R
    """

    data_df = pd.read_csv(fp_whole_exp, delimiter=";")
    data_df["Eye"] = [
        "L" if "left" in fp_protocol else "R" for i in range(len(data_df))
    ]
    data_df = mark_phases(data_df, fp_protocol)

    return data_df


def make_concat_df(fp_recording: list, fp_protocol: str):
    """
    Function for making an experiment dataframe from separate trial files.
    fp_recording - list of strings, list of paths to separate recording files from adaptation and trials, position 0 must be the adaptation
    fp_protocol - string, path to protocol csv file (for eye information)
    Returns:
    concat_df - DataFrame, concatenated trial data with additional column: 'Eye' - dominant eye L or R
    """
    concat_df = pd.concat(
        [pd.read_csv(filepath, delimiter=";") for filepath in fp_recording]
    )
    concat_df["Eye"] = [
        "L" if "left" in fp_protocol else "R" for i in range(len(concat_df))
    ]
    concat_df = mark_phases(concat_df, fp_protocol)
    return concat_df


def load_participant_data(
    participant_no: int,
    data_dir: str,
    include_failed=False,
    save=True,
    save_path="./results/",
):
    """
    Function for loading all recorded data from one participant.
    Input:
    participant_no - int, index of the participant
    data_dir - string, path to directory with participant folders, with directory scheme data_dir/participant_no/03_expsession/retinawise
    include_failed - bool, whether to include the failed runs, if True: they are included, default: False
    save - bool, whether to save the created dataframes, if True, dataframes are saved to folder specified in save_path, default: True
    save_path - string, path to directory for saving the dataframes, default: './results/'
    Returns:
    data_df - DataFrame, includes full recording data with added columns 'Session id' and 'Participant id'
    protocol_timecourse_df - DataFrame, includes timecourse protocol data with added columns 'Session id' and 'Participant id'
    protocol_vars_df - DataFrame, includes protocol variables with added columns 'Session id' and 'Participant id'
    """
    participant_dir = os.path.join(
        data_dir, str(participant_no), "03_expsession\\retinawise"
    )
    protocol_vars_list = []
    protocol_timecourse_list = []
    exp_list = []
    for i, dir in enumerate(os.listdir(participant_dir)):
        rootdir = os.path.join(participant_dir, dir)
        if "test" not in rootdir:
            if include_failed:
                fp_protocol, fp_recording, fp_whole_exp = make_filepaths(rootdir)
                protocol_vars_df, protocol_timecourse_df = make_protocol_dfs(
                    fp_protocol
                )
                exp_df = make_whole_exp_df(fp_whole_exp, fp_protocol)
                protocol_vars_df["Session id"] = [i] * len(protocol_vars_df)
                protocol_vars_df["Participant id"] = [participant_no] * len(
                    protocol_vars_df
                )
                protocol_timecourse_df["Session id"] = [i] * len(protocol_timecourse_df)
                protocol_timecourse_df["Participant id"] = [participant_no] * len(
                    protocol_timecourse_df
                )
                exp_df["Session id"] = [i] * len(exp_df)
                exp_df["Participant id"] = [participant_no] * len(exp_df)
                protocol_vars_list.append(protocol_vars_df)
                protocol_timecourse_list.append(protocol_timecourse_df)
                exp_list.append(exp_df)
            else:
                if "failed" not in rootdir:
                    fp_protocol, fp_recording, fp_whole_exp = make_filepaths(rootdir)
                    protocol_vars_df, protocol_timecourse_df = make_protocol_dfs(
                        fp_protocol
                    )
                    exp_df = make_whole_exp_df(fp_whole_exp, fp_protocol)
                    protocol_vars_df["Session id"] = [i] * len(protocol_vars_df)
                    protocol_vars_df["Participant id"] = [participant_no] * len(
                        protocol_vars_df
                    )
                    protocol_timecourse_df["Session id"] = [i] * len(
                        protocol_timecourse_df
                    )
                    protocol_timecourse_df["Participant id"] = [participant_no] * len(
                        protocol_timecourse_df
                    )
                    exp_df["Session id"] = [i] * len(exp_df)
                    exp_df["Participant id"] = [participant_no] * len(exp_df)
                    protocol_vars_list.append(protocol_vars_df)
                    protocol_timecourse_list.append(protocol_timecourse_df)
                    exp_list.append(exp_df)
                else:
                    continue
        else:
            continue
    data_df = pd.concat(exp_list)
    protocol_vars_df = pd.concat(protocol_vars_list)
    protocol_timecourse_df = pd.concat(protocol_timecourse_list)
    if save:
        data_df.to_csv(save_path + str(participant_no) + "_recording_data.csv")
        protocol_timecourse_df.to_csv(
            save_path + str(participant_no) + "_protocol_timecourse_data.csv"
        )
        protocol_vars_df.to_csv(
            save_path + str(participant_no) + "_protocol_vars_data.csv"
        )
    return data_df, protocol_timecourse_df, protocol_vars_df


class DataLoader:
    pass
