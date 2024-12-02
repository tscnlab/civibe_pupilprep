# civibe-pupilprep

PLEASE READ BEFORE STARTING DATA PROCESSING.

This is a Python repository for preprocessing pupillometry data. The data this was developed for was recorded using the retinaWISE device. Dominant eye was stimulated for the following conditions: flux, mel, lms, l-m, s. The experiment consists of 11 blocks, each with 2 recordings: test a and b, spaced over 40 hours.

The protocol for one recording was as follows:
1. 4 minutes adaptation sequence.
2. 18 seconds sequence composed of: 5 s condition-based stim, 13 s baseline stim.

In each test there was one adaptation sequence, and 5 condition sequences for each condition. The pre-stimulation phase which was used as baseline comes from 1 second period before stimulation onset (taken from the previous sequence).

For the purpose of analysis, using this preprocessing pipeline the data is resampled to 30 Hz and segmented into 19 s long trials, lasting from -1 to 18 s. The artefacts are removed based on non-physiological pupil size outside of 1.5 - 9 mm range (pre-resampling), MAD pupil velocity thresholding (post-resampling), MAD pupil size thresholding (post-resampling, post-velocity thresholding). As the data is already sparse and underwent smoothing in device, we limit artefact removal to minimum.

## Installation

Use pip to install required libraries.

```bash
pip install -r requirements.txt
```

## Notebook tags explanation

load_ - notebooks relating to loading the data from raw data

prep_ - notebooks relating to developing functions for data preprocessing

eda_ - notebooks with exploratory data analysis relating to e.g. exploration of the recording details, determination of thresholds for data acceptance, statistics of data completeness

## Utilities explanation

loading_utils - utilities for loading and marking data from raw files based on protocol

preprocessing_utils - utilities for preprocessing data from dataframes created with loading_utils, as in: resampling, artefact removal, trial/block rejection

visualisation_utils - utilities for plotting data

## Example: load and resample data to 30 Hz (see load_ notebooks)

```python
import sys

sys.path.insert(
    1, "..\\utilities\\"
)  # adds utilities folder to path so we can import modules from it, won't be needed after packaging. in case of Linux - the path connectors (\\) may need to be changed

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

thresholded_df = prep.remove_trials_below_percentage(data_df,baseline_threshold,poi_threshold,baseline_time,poi_time)


```
