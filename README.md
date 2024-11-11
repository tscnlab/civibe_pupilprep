# civibe-pupilprep

This is a Python repository for preprocessing pupillometry data.

## Installation

Use pip to install required libraries.

```bash
pip install -r requirements.txt
```

## Notebook tags explanation

load_ - notebooks relating to loading the data from raw data

prep_ - notebooks relating to data preprocessing

eda_ - notebooks including exploratory data analysis

## Utilities explanation

loading_utils - utilities relating to loading and marking data from raw files

preprocessing_utils - utilities relating to preprocessing data from dataframes created with loading_utils

visualisation_utils - utilities for plotting data

## Example: load and resample data to 50 Hz (see load_ notebooks)

```python
import sys

sys.path.insert(
    1, "..\\utilities\\"
)  # adds utilities folder to path so we can import modules from it, won't be needed after packaging

import loading_utils as load
import preprocessing_utils as prep
import pandas as pd
import numpy as np
import datetime

# make full dataframe for one participant, don't include failed runs, don't save

data_dir = 'path/to/directory/raw/' #path to directory with folders of structure participant/03_expsession/retinawise/...
participant_id = #participant number here
data_df, protocol_timecourse_df, protocol_vars_df = load.load_participant_data(participant_id,data_dir,include_failed=False,save=False)

# make a new df with resampled trials and reduced columns, retaining block information

resampled_df = prep.resample_by_trial(data_df,sample_freq=50)


```

## Example: remove trials with less than 40% not-nan data in baseline and less than 75% not-nan data in signal from 0 to 6 seconds from resampled data for participant 212

```python
import sys

sys.path.insert(
    1, "..\\utilities\\"
)  # adds utilities folder to path so we can import modules from it, won't be needed after packaging

import preprocessing_utils as prep
import pandas as pd
import numpy as np
import datetime

#load resampled dataframe
data_dir = "./results/resampled/" #directory with resampled data 
data_suffix = "_nonan_30_resampled_data.csv" #name of file with 30 Hz resampled data from participant 2xx, name format: 2xxdata_suffix

data_path = os.path.join(data_dir, str(212) + data_suffix)
data_df = pd.read_csv(data_path)

#remove trials below threshold
baseline_threshold = 40
poi_threshold = 75
baseline_time = [-1,0]
poi_time = [0,6]

thresholded_df = remove_trials_below_percentage(data_df,baseline_threshold,poi_threshold,baseline_time,poi_time)


```
