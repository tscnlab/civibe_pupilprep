import sys

sys.path.insert(
    1, "..\\utilities\\"
)  # adds utilities folder to path so we can import modules from it, won't be needed after packaging

import datetime
import pandas as pd
import numpy as np
import os
import loading_utils as load
import preprocessing_utils as prep


def main():
    '''
    For variables' details look in documentation of load.load_participant_data for anything relating to raws, 
    and prep.resample_by_trial for anything relating to resampling
    '''
    participant_list = [200,201,202,204,205,206,207,209,210,211,212,213] # list of participants
    raw_data_dir = "D:/retinawise_mirror/raw/"  #directory to raw data
    include_failed_raw = False   #do you want to include failed runs? 
    save_raw = True   #do you want to save the loaded and marked raw data and protocol files? 
    save_path_raw = './results/'   # save path for raw data, it must exist! add a slash at the end to avoid incorrect saving
    resampling_frequency = 30   #frequency to resample to
    save_path_resampled = './results/resampled/'   #save path for resampled data, must exist!, filenames 2xx_{sampling freq.}_resampled_data.csv

    
    for participant_id in participant_list:
        
        print(str(participant_id)+' :Loading data and resampling to '+str(resampling_frequency)+'...')
        data_df, protocol_timecourse_df, protocol_vars_df = load.load_participant_data(participant_no=participant_id,
                                                                                    data_dir=raw_data_dir,
                                                                                    include_failed=include_failed_raw,
                                                                                    save=save_raw,
                                                                                    save_path=save_path_raw)
        
        data_df = prep.remove_artefacts_non_physio_size(data_df)
        data_df = prep.resample_by_trial(data_df,sample_freq = resampling_frequency)
        
        print('Saving resampled data...')
        resampled_fp = str(participant_id)+'_'+str(resampling_frequency)+'_resampled_data.csv'
        data_df.to_csv(os.path.join(save_path_resampled,resampled_fp))
        print('Done, moving on!')
        
    print('All done!')
        
        
if __name__=='__main__':
    main()



