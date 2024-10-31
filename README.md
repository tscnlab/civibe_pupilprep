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

data_utils - utilities relating to preprocessing and plotting data from dataframes created with loading_utils

## Example: load and resample data (see load_ notebooks)

```python
import sys

sys.path.insert(
    1, "..\\utilities\\"
)  # adds utilities folder to path so we can import modules from it, won't be needed after packaging

import loading_utils
import data_utils
import pandas as pd
import numpy as np

# make full dataframe for one participant, don't include failed runs, don't save

data_dir = 'path/to/directory/raw/' #path to directory with folders of structure participant/03_expsession/retinawise/...
participant_id = #participant number here
data_df, protocol_timecourse_df, protocol_vars_df = loading_utils.load_participant_data(participant_id,data_dir,include_failed=False,save=False)

# make a new df with resampled trials and reduced columns, retaining block information

resampled_df = data_utils.resample_by_trial(data_df)


```

