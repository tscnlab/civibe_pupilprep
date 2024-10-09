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
        if (file.endswith(".csv")) & ("_sine_" in file)
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


def make_protocol_dfs(fp_protocol: str):
    """
    Function for making protocol information dataframes
    fp_protocol - string, path to csv file with protocol data
    Returns:
    protocol_vars_df - DataFrame, includes the first lines with static protocol information
    protocol_timecourse_df - DataFrame, includes the timecourse of the protocol data, with added 'Eye' column - L or R
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
    protocol_timecourse_df["Eye"] = [
        "L" if "left" in fp_protocol else "R"
        for i in range(len(protocol_timecourse_df))
    ]
    return protocol_vars_df, protocol_timecourse_df

def make_whole_exp_df(fp_whole_exp: str, fp_protocol: str):
    """
    Function for making a dataframe from the whole test recording
    fp_whole_exp - string, path to csv file with whole test data
    fp_protocol - string, path to csv with protocol data (for eye information)
    Returns:
    data_df - DataFrame with data from the whole test recording, with an added column 'Eye' - L or R
    """

    data_df = pd.read_csv(fp_whole_exp, delimiter=";")
    data_df["Eye"] = [
        "L" if "left" in fp_protocol else "R" for i in range(len(data_df))
    ]
    return data_df

def make_experiment_df(fp_recording: list, fp_protocol: str):
    """
    Function for making an experiment dataframe from separate trial files.
    fp_recording - list of strings, list of paths to separate recording files from adaptation and trials, position 0 must be the adaptation
    fp_protocol - string, path to protocol csv file (for eye information)
    Returns:
    experiment_df - DataFrame, concatenated trial data with additional columns: 'Trial number' - Adaptation or 1:n, 'Eye' - L or R
    """

    dfs = []
    for i, filepath in enumerate(fp_recording):
        if i == 0:
            data_df = pd.read_csv(filepath, delimiter=";")
            data_df["Trial number"] = ["Adaptation"] * len(data_df)
            dfs.append(data_df)
        else:
            data_df = pd.read_csv(filepath, delimiter=";")
            data_df["Trial number"] = [i] * len(data_df)
            dfs.append(data_df)

    experiment_df = pd.concat(dfs)
    experiment_df["Eye"] = [
        "L" if "left" in fp_protocol else "R" for i in range(len(experiment_df))
    ]
    return experiment_df
