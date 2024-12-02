import datetime
import pandas as pd
import numpy as np
import os
from pupilprep_utilities import preprocessing_utils as prep

def main():
    '''Script for rejecting incomplete trials/conditions in blocks/blocks. 
    '''
    
    participant_list = [200,201,202,204,205,206,207,209,210,211,212,213] 
    save_path_cleaned = './results/cleaned/' #the path with no-artefact data, 2xx_cleaned_data.csv
    save_path_complete = './results/complete/' #the path for saving data after missingness based rejection
    resampling_frequency = 30   #sampling frequency data was resampled to
    baseline_time=[-1,0]   #timeframe for baseline calculation in seconds [start,end]
    poi_time = [0,6]   #timeframe for period of interest in seconds [start,end]
    baseline_threshold = 40   #minimum percentage of completeness in baseline
    poi_threshold = 75   #minimum percentage of completeness in poi
    max_nan_length = 625    #maximum NaN sequence length in miliseconds
    trial_min = 3   #minimum number of trials for a condition in block
    
    for participant_id in participant_list:
        
        print(str(participant_id)+' : Rejecting data...')
        cleaned_fp = str(participant_id)+'_cleaned_data.csv'
        data_df = pd.read_csv(os.path.join(save_path_cleaned,cleaned_fp))
        
        data_df = prep.remove_trials_below_percentage(
                                                        resampled_df=data_df,
                                                        baseline_threshold=baseline_threshold,
                                                        poi_threshold=poi_threshold,
                                                        baseline_time=baseline_time,
                                                        poi_time=poi_time,
                                                    )
        
        data_df = prep.remove_trials_with_long_nans(
                                                    thresholded_df=data_df,
                                                    fs = resampling_frequency,
                                                    max_nan_length = max_nan_length,
                                                    poi_time = poi_time,
                                                    )
        
        data_df = prep.remove_bad_conditions(data_df=data_df, trial_min=trial_min)
        
        data_df = prep.remove_bad_blocks(data_df = data_df)
        
        complete_fp = str(participant_id)+'_complete_data.csv'
        data_df.to_csv(os.path.join(save_path_complete,complete_fp))
        
    print('All done!')

if __name__=='__main__':
    main()