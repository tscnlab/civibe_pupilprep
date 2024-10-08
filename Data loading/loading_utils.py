import pandas as pd
import numpy as np
import seaborn as sns
import os


def make_filepaths(rootdir):
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
    return fp_protocol, fp_recording


def make_protocol_dfs(fp_protocol):
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
    return protocol_vars_df, protocol_timecourse_df


def make_experiment_df(fp_recording):

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
    return experiment_df
