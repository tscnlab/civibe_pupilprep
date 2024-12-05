import datetime
import pandas as pd
import numpy as np
import os
from pupilprep_utilities import pipelines


def main():
    """Script for rejecting incomplete trials/conditions in blocks/blocks in data cleaned of artefacts from save_path_cleaned.
    Saves data fulfilling requirements to save_path_complete.
    For details of functions, look in preprocessing_utils. For pipeline, look in pipelines.
    To change arguments passed into pipeline, modify them in kwargs dictionary below.
    """

    kwargs=dict(participant_list = [200, 201, 202, 204, 205, 206, 207, 209, 210, 211, 212, 213],
    save_path_cleaned = (
        "./results/cleaned/"  
    ),
    save_path_complete = "./results/complete/" ,
    resampling_frequency = 30  ,
    baseline_time = [-1, 0]  ,
    poi_time = [0, 6],  
    baseline_threshold = 40  ,
    poi_threshold = 75 , 
    max_nan_length = 625  ,
    trial_min = 3)  
    
    if not os.path.exists(kwargs['save_path_complete']):
        os.mkdir(kwargs['save_path_complete'])

    for participant_id in kwargs['participant_list']:

        print(str(participant_id) + " : Rejecting data...")
        cleaned_fp = str(participant_id) + "_cleaned_data.csv"
        data_df = pd.read_csv(os.path.join(kwargs['save_path_cleaned'], cleaned_fp))

        data_df = pipelines.reject_incomplete_data(data_df)

        complete_fp = str(participant_id) + "_complete_data.csv"
        data_df.to_csv(os.path.join(kwargs['save_path_complete'], complete_fp))

    print("All done! Data fullfilling completeness requirements saved to "+kwargs['save_path_complete'])


if __name__ == "__main__":
    main()
