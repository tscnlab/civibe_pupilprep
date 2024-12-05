import pandas as pd
import os
from pupilprep_utilities import pipelines


def main():
    """Script for removing artefacts from data loaded from save_path_resampled. Saves to save_path_cleaned.
    Uses velocity thresholding and size thresholding.
    For details of pipelines go to pipelines module.
    For details of functions please go to preprocessing_utils, functions with remove_artifacts prefix.
    If you want to change argument values for pipeline, change them below in kwargs dictionary.
    """
    kwargs=dict(participant_list = [200, 201, 202, 204, 205, 206, 207, 209, 210, 211, 212, 213],
                save_path_resampled = "./results/resampled/" ,
                save_path_cleaned = "./results/cleaned/" ,
                resampling_frequency = 30,
                rolling_window_velocity = 60,
                rolling_window_size = 60,
                multiplier_velocity = 4.5,
                multiplier_size = 4.5,
                columns = [
                    "Stim eye - Size Mm"
                ])
    
    if not os.path.exists(kwargs['save_path_cleaned']):
        os.mkdir(kwargs['save_path_cleaned'])
        
    for participant_id in kwargs['participant_list']:

        print(str(participant_id) + " : Removing artefacts...")
        resampled_fp = (
            str(participant_id)
            + "_"
            + str(kwargs['resampling_frequency'])
            + "_resampled_data.csv"
        )
        data_df = pd.read_csv(os.path.join(kwargs['save_path_resampled'], resampled_fp))

        data_df = pipelines.remove_artefacts(data_df,**kwargs)

        cleaned_fp = str(participant_id) + "_cleaned_data.csv"
        data_df.to_csv(os.path.join(kwargs['save_path_cleaned'], cleaned_fp))

    print("All done! Cleaned data saved to "+kwargs['save_path_cleaned'])


if __name__ == "__main__":
    main()
