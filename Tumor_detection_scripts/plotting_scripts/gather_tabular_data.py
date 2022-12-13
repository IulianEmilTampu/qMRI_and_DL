# %%
"""
Script that gathers the tabular data from the repeated cross-validation trainings.
It returns a .csv file ready for plotting using the implemented plotting utilities.



Steps
1 - gets the necessary paths of where the models are located
2 - for every model, opens the best and last tabular .csv files and stores the
    performance values.
3 - reorganize all the stored values and save summary csv file.
"""
import os
import sys
import csv
import glob
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
        "PATH_TO_MODELS": "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/trained_models_archive/classification_models_10_repeated_cv",
        "SAVE_PATH": "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/trained_models_archive/",
    }

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %% OPEN SUMMARY FILES FOR EVERY MODEL AND STORE DATA
# get list of repeated cross validation folders
CV_REPETITION_FOLDERS = glob.glob(
    os.path.join(args_dict["PATH_TO_MODELS"], "*_cv_repetition_*", "")
)
# get the models within each repetition folder
summary_metrics = []
for r in CV_REPETITION_FOLDERS:
    # get names of models in this folder
    model_names = glob.glob(os.path.join(r, "*", ""))
    model_names = [os.path.basename(os.path.dirname(i)) for i in model_names]
    model_names = ["cMRI", "qMRI", "qMRIGD", "qMRI-qMRIGD"]
    # create dictionary with keys the model names
    aus_dict = {k: [] for k in model_names}
    # populate the dictionary looping through the models
    for m in model_names:
        try:
            aus_dict[m] = pd.read_csv(
                os.path.join(r, m, "tabular_test_summary.csv"), header=0
            ).to_dict()
        except:
            print(
                f"Missing tabular file for input configuration {m}.\nGiven {os.path.join(r, m, 'tabular_test_summary.csv')}\nSaving for this input configuration will be skipped!"
            )
            continue
    summary_metrics.append(aus_dict)
# print summary of the search
print(
    f"Found {len(CV_REPETITION_FOLDERS)} repeated cross validations in the given folder."
)
for idx_r, r_cv in enumerate(summary_metrics):
    print(f"Repetition {idx_r+1}")
    [
        print(f'   {key}: {len(r_cv[key]["classification_type"])/2:} cross validations')
        for key in r_cv.keys()
    ]

# %% AGGREGATE SAVED VALUES and SAVE
header = list(summary_metrics[0][list(summary_metrics[0].keys())[0]].keys())
summary_file = os.path.join(args_dict["SAVE_PATH"], "overall_tabular_test_summary.csv")

if os.path.isfile(summary_file):
    for i in range(100):
        summary_file = os.path.join(
            args_dict["SAVE_PATH"], f"overall_tabular_test_summary_v{i+1}.csv"
        )
        if not os.path.isfile(summary_file):
            break

csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
# build rows to write
rows_to_write = []
# number of cvs per repetition (needed to get the right overall cv number)
nbr_cvs = (
    len(summary_metrics[0][list(summary_metrics[0].keys())[0]]["classification_type"])
    / 2
)
for idx_r, r_cv in enumerate(summary_metrics):
    for model, values in r_cv.items():
        idx_cv = idx_r * nbr_cvs + 1
        for cv in range(len(values[header[0]])):
            aus_row = [values[h][cv] for h in header]
            # replace cv number with the idx_cv
            aus_row[4] = int(idx_cv)
            # add information about the input configuration
            aus_row.insert(3, model)
            # update idx_cv every other cv (handle best last for the same cv)
            if cv % 2 != 0:
                idx_cv += 1
            # save
            rows_to_write.append(aus_row)
# add header for the input type
header.insert(3, "input_type")
# save on file
writer.writerow(header)
writer.writerows(rows_to_write)
csv_file.close()

# %%
