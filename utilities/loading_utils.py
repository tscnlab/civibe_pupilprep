import pandas as pd
import numpy as np
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
    NOW OBSOLETE: look at 'mark_trials' for more precise trial segmentation
    Function for marking phases in the experiment, separately marking transition phases with passive state.
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

def mark_trials(data_df):
    """
    Function for marking phases in the experiment.
    Input:
    data_df - dataframe with all pupillometry data from one participant, needs to have the column 'Eye' for stimulated eye and 'Recording id' for marking separate recordings
    
    Returns:
    data_df - dataframe from input with added columns Trial phase (Adaptation, pre-stim, stim, post-stim), Trial type (type of stimulation), Trial no (from 1::, excludes adaptation), Trial time Sec (from ca. -1 s to end of trial), Stim eye - Size Mm
    """
    data_df["Trial phase"] = ["N/A"] * len(data_df)
    data_df["Trial type"] = ["N/A"] * len(data_df)
    data_df["Trial no"] = ["N/A"] * len(data_df)
    data_df["Trial time Sec"] = ["N/A"] * len(data_df)
    data_df['Stim eye - Size Mm']=["N/A"] * len(data_df)
   
    trial_number = 1
    for session_id in data_df['Recording id'].unique():
        eye = data_df['Eye'][data_df['Recording id']==session_id].unique()[0] #stimulated eye in the session
        for seq_id in sorted(data_df["Sequence index"][data_df['Recording id']==session_id].unique())[1::]: 
            if eye=='L': #block for marking stimulation phase
                data_df.loc[
                    (data_df["Recording id"] == session_id)
                    &(data_df["Sequence index"] == seq_id)
                    & (data_df["Excitation label - Left"] != "baseline")
                    & (data_df["Excitation label - Left"].notna()),
                    "Trial phase",
                ] = "stim"
            else:
                data_df.loc[
                    (data_df["Recording id"] == session_id)
                    &(data_df["Sequence index"] == seq_id)
                    & (data_df["Excitation label - Right"] != "baseline")
                    & (data_df["Excitation label - Right"].notna()),
                    "Trial phase",
                ] = "stim"
                
            if eye =='L': #block for extracting stimulation type in the sequence
                stim = data_df['Excitation label - Left'][(data_df['Sequence index'] == seq_id)
                                                          &(data_df['Recording id'] == session_id) 
                                                          &(data_df['Trial phase']=='stim')].unique()[0]
            else:
                stim = data_df['Excitation label - Right'][(data_df['Sequence index'] == seq_id)
                                                          &(data_df['Recording id'] == session_id) 
                                                          &(data_df['Trial phase']=='stim')].unique()[0]
            
            #block for extracting stim start time and sequence end time in 'whole experiment' time ticks
            seq_start_time = data_df["Sequence time Sec"][
                (data_df["Recording id"] == session_id)
                & (data_df["Sequence index"] == seq_id)
            ].min()
            seq_end_time = data_df["Overall time Sec"][
                (data_df["Recording id"] == session_id)
                & (data_df["Sequence index"] == seq_id)
            ].max()
            stim_start_time = data_df["Overall time Sec"][
                (data_df["Recording id"] == session_id)
                & (data_df["Sequence index"] == seq_id)
                & (data_df["Sequence time Sec"] == seq_start_time)
            ].min()

            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= stim_start_time - 1)
                & (data_df["Overall time Sec"] < stim_start_time),
                "Trial phase",
            ] = "pre-stim" #-1 s from stimulation start time
            
            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= stim_start_time - 1)
                & (data_df["Overall time Sec"] <= seq_end_time),
                "Trial type",
            ] = stim #marks the whole trial type
            
            
            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= stim_start_time - 1)
                & (data_df["Overall time Sec"] <= seq_end_time),
                "Trial no",
            ] = trial_number #marks the trial number in the whole data
            trial_number+=1

            data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= stim_start_time - 1)
                & (data_df["Overall time Sec"] <= seq_end_time),
                "Trial time Sec",
            ] = data_df.loc[
                (data_df["Recording id"] == session_id)
                & (data_df["Overall time Sec"] >= stim_start_time - 1)
                & (data_df["Overall time Sec"] <= seq_end_time),
                "Overall time Sec",
            ]-stim_start_time #marks trial time ticks adjusted to stimulation start time, so that stim starts at 0 and trial starts at -1
            
            
        data_df.loc[
                (data_df["Sequence index"] == 1) & (data_df["Trial phase"] == "N/A"),
                "Trial phase",
            ] = "Adaptation" #marks adaptation phase in sequence 1
        data_df.loc[(data_df["Recording id"] == session_id) & (data_df["Trial phase"] == "N/A"), "Trial phase"] = "post-stim" #marks remaining 'N/A' phases as post-stim
        data_df.loc[(data_df["Recording id"] == session_id) & (data_df["Trial no"]=='N/A') & (data_df["Experiment state"]=='Passive'), "Trial phase"] = "Transition"
    
    data_df.loc[data_df['Eye']=='L','Stim eye - Size Mm'] = data_df['Left - Size Mm']
    data_df.loc[data_df['Eye']=='R','Stim eye - Size Mm'] = data_df['Right - Size Mm']
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
    data_df["Filepath"] = [
        fp_whole_exp for i in range(len(data_df))
    ]

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
    data_df - DataFrame, includes full participant pupillometry data with added columns 'Recording id' and 'Participant id', as well as marked trials as decribed in mark_trials function
    protocol_timecourse_df - DataFrame, includes timecourse protocol data with added columns 'Recording id' and 'Participant id'
    protocol_vars_df - DataFrame, includes protocol variables with added columns 'Recording id' and 'Participant id'
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
                protocol_vars_df["Recording id"] = [i] * len(protocol_vars_df)
                protocol_vars_df["Participant id"] = [participant_no] * len(
                    protocol_vars_df
                )
                protocol_timecourse_df["Recording id"] = [i] * len(protocol_timecourse_df)
                protocol_timecourse_df["Participant id"] = [participant_no] * len(
                    protocol_timecourse_df
                )
                exp_df["Recording id"] = [i] * len(exp_df)
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
                    protocol_vars_df["Recording id"] = [i] * len(protocol_vars_df)
                    protocol_vars_df["Participant id"] = [participant_no] * len(
                        protocol_vars_df
                    )
                    protocol_timecourse_df["Recording id"] = [i] * len(
                        protocol_timecourse_df
                    )
                    protocol_timecourse_df["Participant id"] = [participant_no] * len(
                        protocol_timecourse_df
                    )
                    exp_df["Recording id"] = [i] * len(exp_df)
                    exp_df["Participant id"] = [participant_no] * len(exp_df)
                    protocol_vars_list.append(protocol_vars_df)
                    protocol_timecourse_list.append(protocol_timecourse_df)
                    exp_list.append(exp_df)
                else:
                    continue
        else:
            continue
    data_df = pd.concat(exp_list)
    data_df=mark_trials(data_df)
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
