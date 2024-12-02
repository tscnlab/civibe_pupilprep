import datetime
import pandas as pd
import numpy as np
import os
from pupilprep_utilities import preprocessing_utils as prep

def main():
    '''Script for removing artefacts from data. Uses velocity thresholding and size thresholding.
    For details of variables please go to preprocessing_utils, functions with remove_artifacts prefix.
    '''
    
    participant_list = [200,201,202,204,205,206,207,209,210,211,212,213] 
    save_path_resampled = './results/resampled/'   # the path you used in load_and_resample
    save_path_cleaned = './results/cleaned/' #the path for saving no-artefact data, 2xx_cleaned_data.csv
    resampling_frequency = 30   # frequency data was resampled to
    rolling_window_velocity = 60  # for rolling MAD for velocity thresholding, in samples, for retinawise it's 2 seconds so 2*resampling freq.
    rolling_window_size = 60    # for rolling MAD for size thresholding, in samples, for retinawise it's 2 seconds so 2*resampling freq.
    multiplier_velocity = 4.5   # MAD threshold multiplier for velocity threshold
    multiplier_size = 4.5       # MAD threshold multiplier for size threshold
    columns = ['Stim eye - Size Mm']  # columns to remove artifacts from, by default just stimulated eye
    
    
    for participant_id in participant_list:
        
        print(str(participant_id)+' : Cleaning data...')
        resampled_fp = str(participant_id)+'_'+str(resampling_frequency)+'_resampled_data.csv'
        data_df = pd.read_csv(os.path.join(save_path_resampled,resampled_fp))
        
        for column in columns:
            data_df = prep.remove_artefacts_rolling_velocity_mad(
                            resampled_df = data_df,
                            multiplier = multiplier_velocity,
                            window = rolling_window_velocity,
                            column = column
                        )
            data_df = prep.remove_artefacts_rolling_size_mad(
                            resampled_df = data_df,
                            multiplier = multiplier_size,
                            window = rolling_window_size,
                            column = column
                        )
        
        cleaned_fp = str(participant_id)+'_cleaned_data.csv'
        data_df.to_csv(os.path.join(save_path_cleaned,cleaned_fp))
        
    print('All done!')

if __name__=='__main__':
    main()