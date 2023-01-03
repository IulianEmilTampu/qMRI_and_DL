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
import argparse
import pandas as pd

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
        "PATH_TO_MODELS": "/flush/iulta54/Research/P4-qMRI_git/Tumor_segmentation_scripts/training_scripts/trained_models_archive",
        "SAVE_PATH": "/flush/iulta54/Research/P4-qMRI_git/Tumor_segmentation_scripts/training_scripts/temp_summary_aggregation",
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
    model_names = ["BRATS", "qMRI", "qMRI_GD", "qMRI-qMRI_GD"]
    # create dictionary with keys the model names
    aus_dict = {k: [] for k in model_names}
    # populate the dictionary looping through the models
    for m in model_names:
        try:
            aus_dict[m] = pd.read_csv(
                os.path.join(r, m, "tabular_test_summary.csv"), header=0
            ).to_dict()
        except:
            print(f'Missing info for {os.path.join(r,m,"tabular_test_summary.csv")}')
            aus_dict[m] = None
    summary_metrics.append(aus_dict)

# print summary of the search
print(
    f"Found {len(CV_REPETITION_FOLDERS)} repeated cross validations in the given folder."
)
for idx_r, r_cv in enumerate(summary_metrics):
    print(f"Repetition {idx_r+1}")
    for model in r_cv.keys():
        if r_cv[model] != None:
            print(f'   {model}: {len(r_cv[model]["task"])/2 - 1:} cross validations')
    # [print(f'   {model}: {len(r_cv[model]["classification_type"])/2:} cross validations') for model in r_cv.keys() if r_cv[model] != None]

# %% AGGREGATE SAVED VALUES and SAVE
header = list(summary_metrics[0][list(summary_metrics[0].keys())[0]].keys())

summary_file = os.path.join(args_dict["SAVE_PATH"], "overall_tabular_test_summary.csv")
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
# ausiliary to keep track of the nbr of cv for each model (needed for the overall cv counter)
model_names = ["BRATS", "qMRI", "qMRI_GD", "qMRI-qMRI_GD"]
model_cv_counter = {k: 1 for k in model_names}
# build rows to write
rows_to_write = []
for idx_r, r_cv in enumerate(summary_metrics):
    for model, values in r_cv.items():
        if values != None:
            # save each cv (best and last) for this repetiton and model
            # use the model_cv_counter to set the right overall cv number
            for cv in range(len(values[header[0]])):
                aus_row = [values[h][cv] for h in header]
                # replace cv number with the overall cv number (skip ensamble values)
                if any(
                    [
                        aus_row[header.index("model_version")] == mv
                        for mv in ["best", "last"]
                    ]
                ):
                    if aus_row[header.index("model_version")] == "best":
                        aus_row[header.index("fold")] = model_cv_counter[model]
                    elif aus_row[header.index("model_version")] == "last":
                        # update counter
                        aus_row[header.index("fold")] = model_cv_counter[model]
                        model_cv_counter[model] += 1
                    # add repetition value
                    aus_row.insert(4, idx_r + 1)
                    # save
                    rows_to_write.append(aus_row)

# save also ensemble
for idx_r, r_cv in enumerate(summary_metrics):
    for model, values in r_cv.items():
        if values != None:
            # save each cv (best and last) for this repetiton and model
            for cv in range(len(values[header[0]])):
                aus_row = [values[h][cv] for h in header]
                # replace cv number with the overall cv number (skip ensamble values)
                if aus_row[header.index("fold")] == "ensemble":
                    # add repetition value
                    aus_row.insert(4, idx_r + 1)
                    # save
                    rows_to_write.append(aus_row)

# add repetition to the header
header.insert(4, "cv_repetition")
# save on file
writer.writerow(header)
writer.writerows(rows_to_write)
csv_file.close()

# %%
