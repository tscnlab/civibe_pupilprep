import pandas as pd
import numpy as np
import os


def make_filepaths(rootdir: str):
    """
    Function for extracting paths to protocol and data files
    rootdir - string, path to directory with subject experiment files
    Returns:
    fp_protocol - string, path to protocol csv file
    fp_sequence - list of strings, paths to separate adaptation/trial recording csv files
    fp_whole_exp - string, path to the whole test csv recording
    """
    fp_protocol = [
        os.path.join(subdirs, file)
        for subdirs, dirs, files in os.walk(rootdir)
        for file in files
        if (file.endswith(".csv")) & ("sine" in file)
    ][0]

    fp_sequence = [
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

    return fp_protocol, fp_sequence, fp_whole_exp


def mark_trials(data_df:pd.DataFrame, delay:float=1.5):
    """
    Function for marking phases in the experiment, accounting for software delay. 
    Warning: only columns created here have corrected labelling in regards to delay. 
    The raw data columns for excitation label, excitation index etc. are incorrect and should not be used further, since the labels were interpolated from time elapsed without accounting for delay.
    Input:
    data_df - dataframe with all pupillometry data from one participant, needs to have the column 'Eye' for stimulated eye and 'Recording id' for marking separate recordings
    delay - software-hardware delay in seconds (actual stimulation starts at time 0+delay)
    Returns:
    data_df - dataframe from input with added columns Trial phase (Adaptation, pre-stim, stim, post-stim), Trial type (type of stimulation), Trial no (from 1::, excludes adaptation), Trial time Sec (from ca. -1 s to end of trial), Stim eye - Size Mm
    """
    data_df["Trial phase"] = ["N/A"] * len(data_df) #pre-stim, stim, post-stim
    data_df["Trial type"] = ["N/A"] * len(data_df) #s,lms,l-m,flux,mel
    data_df["Trial no"] = pd.Series() #only for trials with stimulation 1::
    data_df["Trial time Sec"] = pd.Series() #-1:
    data_df["Stim eye - Size Mm"] = pd.Series()

    trial_number = 1
    for session_id in data_df["Recording id"].unique():
        eye = data_df["Eye"][data_df["Recording id"] == session_id].unique()[
            0
        ]  # stimulated eye in the recording
        for seq_id in sorted(
            data_df["Sequence index"][data_df["Recording id"] == session_id].unique()
        )[1::]:
            # block for extracting stim start time and sequence end time in 'whole experiment' time ticks
            corrected_seq_start_time = data_df["Sequence time Sec"][
                (data_df["Recording id"] == session_id)
                & (data_df["Sequence index"] == seq_id)
            ].min() + delay #get true sequence start time by taking minimum of seq time column and adding delay 
            
            corrected_start_time = data_df["Overall time Sec"][
                (data_df["Recording id"] == session_id)
                & (data_df["Sequence index"] == seq_id)
                & (data_df["Sequence time Sec"] >=corrected_seq_start_time)
            ].min() #we take the overall time value corresponding to minimum of time values larger or equal to corrected sequence start time (since it may not be exact due to uneven sampling)
            
            seq_end_time = data_df["Overall time Sec"][
                (data_df["Recording id"] == session_id)
                & (data_df["Sequence index"] == seq_id)
                & (data_df["Experiment state"] == 'Active')
            ].max() #max overall time value for this sequence for active exp state
            
            
            # block for marking pre-stim and stim phases
            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= corrected_start_time - 1)
                & (data_df["Overall time Sec"] < corrected_start_time),
                "Trial phase",
            ] = "pre-stim"  # -1 s from stimulation start time
            
            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= corrected_start_time)
                & (data_df["Overall time Sec"] < corrected_start_time+5),
                "Trial phase",
            ] = "stim"  # 5 s of stimulation

            # block for extracting stimulation type in the sequence and marking the trial stimulation type
            if eye == "L":  
                stim = data_df["Excitation label - Left"][
                    (data_df["Sequence index"] == seq_id)
                    & (data_df["Recording id"] == session_id)
                    & (data_df["Trial phase"] == "stim")
                    & (data_df["Excitation label - Left"].isin(['s','lms','l-m','flux','mel']))
                ].unique()[0]
            else:
                stim = data_df["Excitation label - Right"][
                    (data_df["Sequence index"] == seq_id)
                    & (data_df["Recording id"] == session_id)
                    & (data_df["Trial phase"] == "stim")
                    & (data_df["Excitation label - Right"].isin(['s','lms','l-m','flux','mel']))
                ].unique()[0]
                
            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= corrected_start_time - 1)
                & (data_df["Overall time Sec"] <= seq_end_time),
                "Trial type",
            ] = stim  # marks the whole trial as stim type (condition)

            # mark trial number
            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= corrected_start_time - 1)
                & (data_df["Overall time Sec"] <= seq_end_time),
                "Trial no",
            ] = trial_number  
            trial_number += 1

            # calculate trial time from -1 to end of sequence, with start of stim being at 0
            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= corrected_start_time - 1)
                & (data_df["Overall time Sec"] <= seq_end_time),
                "Trial time Sec",
            ] = (
                data_df.loc[
                    (data_df["Recording id"] == session_id)
                    & (data_df["Overall time Sec"] >= corrected_start_time - 1)
                    & (data_df["Overall time Sec"] <= seq_end_time),
                    "Overall time Sec",
                ]
                - corrected_start_time
            )  # marks trial time ticks adjusted to stimulation start time, so that stim starts around 0 and trial starts at -1

        data_df.loc[
            (data_df["Sequence index"] == 1) & (data_df["Trial phase"] == "N/A"),
            "Trial phase",
        ] = "Adaptation"  # marks adaptation phase in sequence 1
        data_df.loc[
            (data_df["Recording id"] == session_id) & (data_df["Trial phase"] == "N/A"),
            "Trial phase",
        ] = "post-stim"  # marks remaining 'N/A' phases as post-stim
        data_df.loc[
            (data_df["Recording id"] == session_id)
            & (data_df["Trial no"].isna())
            & (data_df["Experiment state"] == "Passive"),
            "Trial phase",
        ] = "Transition" # changes marking to Transition if experiment state is passive and doesn't belong to a trial

    data_df.loc[data_df["Eye"] == "L", "Stim eye - Size Mm"] = data_df["Left - Size Mm"]
    data_df.loc[data_df["Eye"] == "R", "Stim eye - Size Mm"] = data_df["Right - Size Mm"]
    
    return data_df


def make_protocol_dfs(fp_protocol: str):
    """
    Function for making protocol information dataframes
    fp_protocol - string, path to csv file with protocol data
    Returns:
    protocol_vars_df - DataFrame, includes the first lines with static protocol information plus information on stimulated eye
    protocol_timecourse_df - DataFrame, includes the timecourse of the protocol data, with added 'Eye' column - stimulated eye L or R
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
        protocol_vars_df.loc[len(protocol_vars_df)] = ["Stimulated eye", "L", np.nan]
    else:
        protocol_timecourse_df["Eye"] = [
            "R" for i in range(len(protocol_timecourse_df))
        ]
        protocol_vars_df.loc[len(protocol_vars_df)] = ["Stimulated eye", "R", np.nan]

    return protocol_vars_df, protocol_timecourse_df


def make_whole_exp_df(fp_whole_exp: str, fp_protocol: str):
    """
    Function for making a dataframe from the whole test recording
    fp_whole_exp - string, path to csv file with whole test data
    fp_protocol - string, path to csv with protocol data (for eye information)
    Returns:
    data_df - DataFrame with data from the whole test recording, with an added column 'Eye' - L or R, "Filepath" - path to data file
    """

    data_df = pd.read_csv(fp_whole_exp, delimiter=";")
    data_df["Eye"] = [
        "L" if "left" in fp_protocol else "R" for i in range(len(data_df))
    ]
    data_df["Filepath"] = [fp_whole_exp for i in range(len(data_df))]
    
    return data_df


def make_concat_df(fp_sequence: list, fp_protocol: str):
    """
    Function for making an experiment dataframe from separate trial files.
    fp_sequence - list of strings, list of paths to separate recording files from adaptation and trials, position 0 must be the adaptation
    fp_protocol - string, path to protocol csv file (for eye information)
    Returns:
    concat_df - DataFrame, concatenated trial data with additional column: 'Eye' - dominant eye L or R
    """
    concat_df = pd.concat(
        [pd.read_csv(filepath, delimiter=";") for filepath in fp_sequence]
    )
    concat_df["Eye"] = [
        "L" if "left" in fp_protocol else "R" for i in range(len(concat_df))
    ]

    return concat_df


def load_participant_data(
    participant_no: int,
    data_dir: str,
    delay: float=1.0,
    include_failed=False,
    save=True,
    save_path="./results/",
):
    """
    Function for loading all recorded data from one participant.
    Input:
    participant_no - int, index of the participant
    data_dir - string, path to directory with participant folders, with directory scheme data_dir/participant_no/03_expsession/retinawise
    delay - float, delay in seconds from SW-HW, default:1.0
    include_failed - bool, whether to include the failed runs, if True: they are included, default: False
    save - bool, whether to save the created dataframes, if True, dataframes are saved to folder specified in save_path, default: True
    save_path - string, path to directory for saving the dataframes, default: './results/'
    Returns:
    data_df - DataFrame, includes full participant pupillometry data with added columns 'Recording id' and 'Participant id', as well as marked trials as decribed in mark_trials function
    protocol_timecourse_df - DataFrame, includes timecourse protocol data with added columns 'Recording id' and 'Participant id'
    protocol_vars_df - DataFrame, includes protocol variables with added columns 'Recording id' and 'Participant id'
    """
    participant_dir = os.path.join(
        data_dir, str(participant_no), "03_expsession/retinawise"
    )
    protocol_vars_list = []
    protocol_timecourse_list = []
    exp_list = []
    # iterate over directories in participants directory
    for i, dir in enumerate(sorted(os.listdir(participant_dir))):
        rootdir = os.path.join(participant_dir, dir)
        # condition to exclude the folders that were test runs
        if "test" not in rootdir:
            # block for loading all data, including failed runs
            if include_failed:
                fp_protocol, fp_sequence, fp_whole_exp = make_filepaths(rootdir)
                protocol_vars_df, protocol_timecourse_df = make_protocol_dfs(
                    fp_protocol
                )
                exp_df = make_whole_exp_df(fp_whole_exp, fp_protocol)
                
                # get block and test numbers from directory path
                if 'failed' in rootdir:
                    if ("10a" in rootdir) or ("10b" in rootdir): #block 10 is an irregularity
                        block = 10
                        test = rootdir[-8] #suffix _failed is at the end of the dir name, we need to get element at -8th position to get test
                    else:
                        block = int(rootdir[-9]) #as above, folders have name format xxx_9a_failed
                        test = rootdir[-8]
                else:
                    if ("10a" in rootdir) or ("10b" in rootdir):
                        block = 10
                        test = rootdir[-1]
                    else:
                        block = int(rootdir[-2])
                        test = rootdir[-1]
                
                # assign values in protocol and recording dataframes denoting recording, participant, block and test
                protocol_vars_df["Recording id"] = [i] * len(protocol_vars_df)
                protocol_vars_df["Participant id"] = [participant_no] * len(
                    protocol_vars_df
                )
                
                protocol_vars_df["Block"] = [block] * len(protocol_vars_df)
                protocol_vars_df["Test"] = [test] * len(protocol_vars_df)
                
                protocol_timecourse_df["Recording id"] = [i] * len(
                    protocol_timecourse_df
                )
                protocol_timecourse_df["Participant id"] = [participant_no] * len(
                    protocol_timecourse_df
                )
                protocol_timecourse_df["Block"] = [block] * len(
                        protocol_timecourse_df
                    )
                protocol_timecourse_df["Test"] = [test] * len(
                    protocol_timecourse_df
                )
                exp_df["Recording id"] = [i] * len(exp_df)
                exp_df["Participant id"] = [participant_no] * len(exp_df)
                exp_df["Block"] = [block] * len(exp_df)
                exp_df["Test"] = [test] * len(exp_df)
                
                # save dataframes from one recording to lists to concatenate them later
                protocol_vars_list.append(protocol_vars_df)
                protocol_timecourse_list.append(protocol_timecourse_df)
                exp_list.append(exp_df)
            # block for loading only not failed data
            else:
                if "failed" not in rootdir:

                    fp_protocol, fp_recording, fp_whole_exp = make_filepaths(rootdir)
                    protocol_vars_df, protocol_timecourse_df = make_protocol_dfs(
                        fp_protocol
                    )
                    exp_df = make_whole_exp_df(fp_whole_exp, fp_protocol)

                    # get block and test names
                    if ("10a" in rootdir) or ("10b" in rootdir): #block 10 is an irregularity
                        block = 10
                        test = rootdir[-1]
                    else:
                        block = int(rootdir[-2])
                        test = rootdir[-1]

                    # assign values in dataframes denoting recording, participant, block and test numbers
                    protocol_vars_df["Recording id"] = [i] * len(protocol_vars_df)
                    protocol_vars_df["Participant id"] = [participant_no] * len(
                        protocol_vars_df
                    )
                    protocol_vars_df["Block"] = [block] * len(protocol_vars_df)
                    protocol_vars_df["Test"] = [test] * len(protocol_vars_df)
                    protocol_timecourse_df["Recording id"] = [i] * len(
                        protocol_timecourse_df
                    )
                    protocol_timecourse_df["Participant id"] = [participant_no] * len(
                        protocol_timecourse_df
                    )
                    protocol_timecourse_df["Block"] = [block] * len(
                        protocol_timecourse_df
                    )
                    protocol_timecourse_df["Test"] = [test] * len(
                        protocol_timecourse_df
                    )

                    exp_df["Recording id"] = [i] * len(exp_df)
                    exp_df["Participant id"] = [participant_no] * len(exp_df)
                    exp_df["Block"] = [block] * len(exp_df)
                    exp_df["Test"] = [test] * len(exp_df)
                    protocol_vars_list.append(protocol_vars_df)
                    protocol_timecourse_list.append(protocol_timecourse_df)
                    exp_list.append(exp_df)
                else:
                    continue
        else:
            continue
    data_df = pd.concat(exp_list)
    data_df.reset_index(inplace=True)
    data_df = mark_trials(data_df,delay)
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
