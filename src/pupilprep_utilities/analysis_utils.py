import pandas as pd
import numpy as np
import os
import pupilprep_utilities.loading_utils as load
import pupilprep_utilities.preprocessing_utils as prep


def make_completeness_stats_df(
    data_dir: str,
    data_suffix: str = "_complete_data.csv",
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
    blocks: list = range(0, 11),
    conditions: list = ["flux", "l-m", "lms", "mel", "s"],
):
    """Function that generates a completeness dataframe for all participants.
    Has information on how many trials fulfill requirements of completeness in blocks.

    Args:
        data_dir (str): directory to data to analyse.
        data_suffix (str): suffix of datafiles to analyse. Format: 200data_suffix. Defaults to '_complete_data.csv'.
        participant_list (list, optional): List of participants. Defaults to [200, 201, 202, 204, 205, 206, 207, 209, 210, 211, 212, 213].
        blocks (list, optional): List of block numbers. Defaults to range(0, 11).
        conditions (list, optional): List of condition names. Defaults to ["flux", "l-m", "lms", "mel", "s"].

    Returns:
        pd.DataFrame: dataframe with information on how many trials are available per condition in blocks. If less than 3, it's marked as such.
    """
    completeness_dict = {
        "Participant": [],
        "Block": [],
        "Condition": [],
        "Trial count": [],
        "Block available": [],
    }

    for participant_id in participant_list:
        data_path = os.path.join(data_dir, str(participant_id) + data_suffix)
        data_df = pd.read_csv(data_path)
        groupby_df = (
            data_df[["Block", "Trial type", "Trial no"]]
            .groupby(["Block", "Trial type"])
            .agg("nunique")
        )

        groupby_df.reset_index(inplace=True)

        for block in blocks:
            for condition in conditions:
                if block in groupby_df["Block"].values:

                    if (
                        condition
                        in groupby_df["Trial type"][groupby_df["Block"] == block].values
                    ):
                        count = groupby_df["Trial no"][
                            (groupby_df["Block"] == block)
                            & (groupby_df["Trial type"] == condition)
                        ].values[0]
                        block_acc = "yes"
                    else:
                        count = "less than 3"
                        block_acc = "no"
                else:
                    block_acc = "no"
                    count = "less than 3"
                completeness_dict["Participant"].append(participant_id)
                completeness_dict["Block"].append(block)
                completeness_dict["Condition"].append(condition)
                completeness_dict["Trial count"].append(count)
                completeness_dict["Block available"].append(block_acc)

    completeness_df = pd.DataFrame(completeness_dict)
    return completeness_df
