import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_change_from_baseline(data_df):
    data_df['Baseline change %'] = [pd.NA]*len(data_df)
    for i in data_df['Trial no'].unique()[1::]:
        trial_df = data_df[(data_df['Trial no']==i)].copy()
        baseline = trial_df['Stim eye - Size Mm'][trial_df['Trial time Sec']<0].mean()
        data_df.loc[(data_df['Trial no']==i),'Baseline change %'] = (trial_df['Stim eye - Size Mm'] - baseline)*100/baseline
    return data_df


def plot_trials(data_df,participant_id, trial_types: list):  
    fig,axs = plt.subplots(1,len(trial_types),figsize=(len(trial_types)*5,5))
    for i,trial_type in enumerate(trial_types):
        axs[i].axvspan(0, 5, alpha=0.1, color='green')
        sns.scatterplot(ax=axs[i],data = data_df[data_df['Trial type']==trial_type],x='Trial time Sec',y='Stim eye - Size Mm',hue='Trial no',s=0.5)
        axs[i].set_title(trial_type)
        if i>0:
            axs[i].set_ylabel('')
        axs[i].legend([],[], frameon=False)
        axs[i].set_ylim([0,np.max(data_df['Stim eye - Size Mm'][data_df['Trial type']==trial_type])+1])
    fig.suptitle(f'Pupil size per trial for participant {participant_id}')
    plt.tight_layout()
    plt.show()
    return fig,axs


def plot_baseline_change(data_df,participant_id,trial_types: list):
    fig,axs = plt.subplots(1,len(trial_types),figsize=(len(trial_types)*5,5))
    for i,trial_type in enumerate(trial_types):
        axs[i].axvspan(0, 5, alpha=0.1, color='green')
        sns.scatterplot(ax=axs[i],data = data_df[data_df['Trial type']==trial_type],x='Trial time Sec',y='Baseline change %',hue='Trial no',s=0.5)
        axs[i].set_title(trial_type)
        if i>0:
            axs[i].set_ylabel('')
        axs[i].legend([],[], frameon=False)
        axs[i].set_ylim([np.min(data_df['Baseline change %'][data_df['Trial type']==trial_type])-1,
                         np.max(data_df['Baseline change %'][data_df['Trial type']==trial_type])+1])
    fig.suptitle(f'Change in stimulated pupil size from baseline average per trial for participant {participant_id}')
    plt.tight_layout()
    plt.show()
    return fig,axs
