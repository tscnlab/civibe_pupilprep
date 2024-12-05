import pandas as pd
import os
from pupilprep_utilities import pipelines


def main():
    """
    Script for loading and resampling data. Saves resampled data to save_path_resampled. Optionally saves raw data to save_path_raw.
    For variables' details look in documentation of load.load_participant_data for anything relating to raws,
    and prep.resample_by_trial for anything relating to resampling. 
    For details of pipelines go to pipelines module.
    If you want to change argument values passed to pipeline, change them below in kwargs dictionary.
    """
    kwargs=dict(raw_data_dir = "D:/retinawise_mirror/raw/" ,
                participant_list = [200, 201, 202, 204, 205, 206, 207, 209, 210, 211, 212, 213],
                include_failed_raw = False,
                save_raw=True,
                save_path_raw="./results/raw/",
                delay_raw = 0,
                resampling_frequency = 30,
                save_path_resampled = "./results/resampled/"
    )

    if not os.path.exists(kwargs['save_path_resampled']):
        os.mkdir(kwargs['save_path_resampled'])
    if kwargs['save_raw']:
        if not os.path.exists(kwargs['save_path_raw']):
            os.mkdir(kwargs['save_path_raw'])
            
    for participant_id in kwargs['participant_list']:

        print(
            str(participant_id)
            + " : Loading data and resampling to "
            + str(kwargs['resampling_frequency'])
            + "..."
        )
        
        data_df = pipelines.load_and_resample(kwargs['raw_data_dir'],participant_id,**kwargs)

        resampled_fp = (
            str(participant_id)
            + "_"
            + str(kwargs['resampling_frequency'])
            + "_resampled_data.csv"
        )
        
        data_df.to_csv(os.path.join(kwargs['save_path_resampled'], resampled_fp))

    print("All done! Resampled data saved to "+kwargs['save_path_resampled'])
    if kwargs['save_raw']:
        print('Raw data saved to '+kwargs['save_path_raw'])


if __name__ == "__main__":
    main()
