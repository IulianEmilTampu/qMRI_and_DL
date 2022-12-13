# %%
"""
Script that gathers the logits data from the repeated cross-validation trainings.
It returns a .json file ready for plotting using the implemented plotting utilities 
(plot_ROC_comparison_from_summary_file.py).

Steps
1 - get the necessary paths of where the models are located
2 - for every model, opens the best and last tabular .csv files and stores the
    performance values.
3 - reorganize all the stored values and save summary csv file.
"""
import os
import sys
import csv
import glob
import json
import numpy as np
import pandas as pd
import argparse

# %%
su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Script that gathers the logits data from the repeated cross-validation trainings."
    )
    parser.add_argument(
        "-ptm",
        "--PATH_TO_MODELS",
        required=True,
        help="Path to where the folder containing the repeated cross validation models are saved.",
    )
    parser.add_argument(
        "-sp",
        "--SAVE_PATH",
        required=True,
        help="Provide the path where to save the summary .json file of the gathered information.",
    )

    args_dict = dict(vars(parser.parse_args()))
else:
    print("Running in debug mode.")
    args_dict = {
        "PATH_TO_MODELS": "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/training_scripts/trained_models_archive",
        "SAVE_PATH": "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/training_scripts/temp_summary_aggregation",
    }

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %% OPEN LOGITS SUMMARY FILES FOR EVERY MODEL AND STORE DATA
"""
The data is strucrute as follows
INPUT_CONFIGURATION
    MODEL_VESRION : 'best', 'last'
        FOLD_NBR 
            test_prediction : list
            test_softmax : list
"""

CV_REPETITION_FOLDERS = glob.glob(
    os.path.join(args_dict["PATH_TO_MODELS"], "*_cv_repetition_*", "")
)
INPUT_CONFIG = ["cMRI", "qMRI", "qMRIGD", "qMRI-qMRIGD"]
# where to save data
summary_dict = {}
for ic in INPUT_CONFIG:
    summary_dict[ic] = {"best": {}, "last": {}}

# keep track of fold numbers
model_cv_counter = {k: 1 for k in INPUT_CONFIG}

for r in CV_REPETITION_FOLDERS:
    for ic in INPUT_CONFIG:
        # open summary file for this repetition and input configuration
        summary_file = os.path.join(r, ic, "summary_test_logits.json")
        try:
            with open(summary_file, "r") as file:
                summary_file_data = json.load(file)

            # save information in the right place
            for f in range(len(summary_file_data)):
                for model_version in ["best", "last"]:
                    summary_dict[ic][model_version][model_cv_counter[ic]] = {
                        "per_sample_pred": summary_file_data[str(f + 1)][model_version][
                            "per_sample_pred"
                        ],
                        "per_sample_logits": summary_file_data[str(f + 1)][
                            model_version
                        ]["per_sample_logits"],
                        "per_sample_ground_truth": summary_file_data[str(f + 1)][
                            model_version
                        ]["per_sample_ground_truth"],
                    }
                # update fold counter
                model_cv_counter[ic] += 1
        except:
            print(
                f"Missing summary file for input configuration {ic}.\nGiven {summary_file}\nSaving for this input configuration will be skipped!"
            )
            continue

# %% SAVE VALUES
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


summary_file = os.path.join(args_dict["SAVE_PATH"], f"gathered_test_logits.json")
if os.path.isfile(summary_file):
    for i in range(100):
        summary_file = os.path.join(
            args_dict["SAVE_PATH"], f"gathered_test_logits_v{i+1}.json"
        )
        if not os.path.isfile(summary_file):
            break

with open(summary_file, "w") as fp:
    json.dump(summary_dict, fp, cls=NpEncoder)

# %%
