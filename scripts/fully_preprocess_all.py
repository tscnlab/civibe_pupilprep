import pandas as pd
import os
from pupilprep_utilities import pipelines


def main():
    """Program performing the full preprocessing pipeline on all participants. By default saves intermediate steps.
    If you want to change arguments for any step of the pipeline, add them to kwargs dictionary below.
    For details of available variables, go to pipelines module.
    """
    kwargs = dict(
        participant_list=[200, 201, 202, 204, 205, 206, 207, 209, 210, 211, 212, 213],
        raw_data_dir="D:/retinawise_mirror/raw/",
        save_intermediate_steps=True,
        save_path_raw="./results/raw/",
        save_path_resampled="./results/resampled/",
        save_path_cleaned="./results/cleaned/",
        save_path_complete="./results/complete/",
        save_path_final="./results/final/",
    )

    if kwargs["save_intermediate_steps"]:
        for key in kwargs.keys():
            if "path" in key:
                if not os.path.exists(kwargs[key]):
                    os.makedirs(kwargs[key])
    else:
        if not os.path.exists(kwargs["save_path_final"]):
            os.makedirs(kwargs["save_path_final"])

    for participant_id in kwargs["participant_list"]:
        print(str(participant_id) + ": performing preprocessing pipeline.")
        data_df = pipelines.full_preprocessing_pipeline(
            participant_id, **kwargs
        )
        data_df.to_csv(
            os.path.join(
                kwargs["save_path_final"], str(participant_id) + "_final_data.csv"
            )
        )

    print("All done! Final data saved to " + kwargs["save_path_final"])

    if kwargs["save_intermediate_steps"] == True:
        print("Raw data saved to " + kwargs["save_path_raw"])
        print("Resampled data saved to " + kwargs["save_path_resampled"])
        print("Cleaned data saved to " + kwargs["save_path_cleaned"])
        print(
            "Data fulfilling completeness reqs saved to " + kwargs["save_path_complete"]
        )


if __name__ == "__main__":
    main()
