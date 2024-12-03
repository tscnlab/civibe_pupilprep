# civibe-pupilprep

PLEASE READ BEFORE STARTING DATA PROCESSING.

This is a Python repository for preprocessing pupillometry data. It is licensed under GPL-3.

The data this was developed for was recorded using the retinaWISE device. Dominant eye was stimulated for the following conditions: flux, mel, lms, l-m, s. The experiment consists of 11 blocks, each with 2 recordings: test a and b, spaced over 40 hours.

The protocol for one recording was as follows:
1. 4 minutes adaptation sequence.
2. 18 seconds sequence composed of: 5 s condition-based stim, 13 s baseline stim.

In each test there was one adaptation sequence, and 5 condition sequences for each condition. The pre-stimulation phase which was used as baseline comes from 1 second period before stimulation onset (taken from the previous sequence).

For the purpose of analysis, using this preprocessing pipeline the data is resampled to 30 Hz and segmented into 19 s long trials, lasting from -1 to 18 s. The artefacts are removed based on non-physiological pupil size outside of 1.5 - 9 mm range (pre-resampling), MAD pupil velocity thresholding (post-resampling), MAD pupil size thresholding (post-resampling, post-velocity thresholding). As the data is already sparse and underwent smoothing in device, we limit artefact removal to minimum.

## Installation

First you need to build a wheel for the package. Run this in the repository folder:

```bash
python setup.py bdist_wheel
```
Then install the package from the wheel. To do so, run:

```bash
pip install .\dist\wheel-name-here.whl
```
For instance, for version 0.1.1 the wheel name is civibe_pupilprep_utils-0.1.1-py3-none-any.whl. Adjust formatting as needed (the above is for Windows). The package and dependencies should now be installed.

Then you can import the package for instance as:

```python

import pupilprep_utilities

print(pupilprep_utilities.preprocessing_utils.resample_by_trial.__doc__)
```

or better import modules you need as:

```python

import pupilprep_utilities.preprocessing_utils as prep

print(prep.resample_by_trial.__doc__)
```

## Scripts to run before running the notebooks

1. Run load_and_resample.py to get raw and resampled data. By default, data is resampled to 30 Hz and cut to -1:18 s trial segments.
2. Run remove_artefacts.py to perform artefact removal on the resampled data. By default, it performs thresholding based on rolling pupil velocity MAD and rolling pupil size MAD.
3. Run reject_incomplete.py to perform trial/block rejection on resampled and cleaned/ or just resampled data. By default the period of interest is 0:6 s with min. 75% completeness, baseline is -1:0 s with min. 40% completeness, no long NaN > 625 ms. A block is valid for the condition if it has min. 3 trials for that condition and flux. 

## Notebooks explanation

load_ - notebooks relating to loading the data from raw experiment files

prep_ - notebooks relating to developing functions for data preprocessing

eda_ - notebooks with exploratory data analysis relating to e.g. exploration of the recording details, determination of thresholds for data acceptance, statistics of data completeness

All notebooks have a section 'Purpose of the notebook' which explains which data they use, from which script.

## Utilities explanation

loading_utils - utilities for loading and marking data from raw files based on protocol

preprocessing_utils - utilities for preprocessing data from dataframes created with loading_utils, as in: resampling, artefact removal, trial/block rejection

visualisation_utils - utilities for plotting data

