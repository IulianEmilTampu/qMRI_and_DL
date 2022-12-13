# %%
"""
Script that given a model and the configuration file used for training, tests
the model on the test dataset and save the information in a txt file

Steps
1 - get paths for the image data and the model
2 - for every fold, load the last and best models
3 - test models and save fold information
"""

import os
import sys
import glob
import json
import cv2
import csv
import warnings
import numpy as np
import argparse
import importlib
from pathlib import Path
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning)

# local imports
import utilities

# %%
su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Test routine for models trained in the context of Tumor detection on qMRI data"
    )
    parser.add_argument(
        "-wf",
        "--WORKING_FOLDER",
        required=True,
        help="Provide the working folder where the trained model will be saved.",
    )
    parser.add_argument(
        "-df",
        "--IMG_DATASET_FOLDER",
        required=True,
        help="Provide the Image Dataset Folder where the folders for each modality are located.",
    )
    parser.add_argument(
        "-af",
        "--ANNOTATION_DATASET_FOLDER",
        required=True,
        help="Provide the Annotation  Folder where annotations are located.",
    )
    parser.add_argument(
        "-bmf",
        "--BRAIN_MASK_FOLDER",
        required=True,
        help="Provide the Brain Mask Folder where annotations are located.",
    )
    parser.add_argument(
        "-mp",
        "--MODEL_PATH",
        required=True,
        help="Path to the model to test. This is the folder where the cross validation folders are located.",
    )
    parser.add_argument(
        "-tvt",
        "--TRAIN_VALIDATION_TEST_FILE",
        required=True,
        help="Path to the .json file where it is specified which subjects to use for testing",
    )
    parser.add_argument(
        "-g",
        "--GPU_NBR",
        required=False,
        default=0,
        help="Provide the GPU number to use for testing.",
    )

    args_dict = dict(vars(parser.parse_args()))
    args_dict["NBR_FOLDS"] = len(
        glob.glob(os.path.join(args_dict["MODEL_PATH"], "fold_*", ""))
    )

else:
    # # # # # # # # # # # # # # DEBUG
    print("Running in debug mode.")
    args_dict = {}
    args_dict[
        "WORKING_FOLDER"
    ] = "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/training_scripts"
    args_dict[
        "IMG_DATASET_FOLDER"
    ] = "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality"
    args_dict[
        "ANNOTATION_DATASET_FOLDER"
    ] = "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality/All_Annotations"
    args_dict[
        "BRAIN_MASK_FOLDER"
    ] = "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality/All_Brain_masks"
    args_dict[
        "TRAIN_VALIDATION_TEST_FILE"
    ] = "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/training_scripts/trained_models_archive/Test/train_val_test_subjects.json"
    args_dict[
        "MODEL_PATH"
    ] = "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/training_scripts/trained_models_archive/Test/cMRI"
    args_dict["GPU_NBR"] = "0"
    args_dict["NBR_FOLDS"] = len(
        glob.glob(os.path.join(args_dict["MODEL_PATH"], "fold_*", ""))
    )

combination_dict = {
    "0": ["cMRI", ["All_T1FLAIR_GD", "All_T1FLAIR", "All_T2FLAIR", "All_T2"]],
    "1": ["qMRI", ["All_qMRI_T1", "All_qMRI_T2", "All_qMRI_PD"]],
    "2": ["qMRIGD", ["All_qMRI_T1_GD", "All_qMRI_T2_GD", "All_qMRI_PD_GD"]],
    "3": [
        "cMRI-qMRI",
        [
            "All_T1FLAIR_GD",
            "All_T1FLAIR",
            "All_T2FLAIR",
            "All_T2",
            "All_qMRI_T1",
            "All_qMRI_T2",
            "All_qMRI_PD",
        ],
    ],
    "4": [
        "cMRI-qMRIGD",
        [
            "All_T1FLAIR_GD",
            "All_T1FLAIR",
            "All_T2FLAIR",
            "All_T2",
            "All_qMRI_T1_GD",
            "All_qMRI_T2_GD",
            "All_qMRI_PD_GD",
        ],
    ],
    "5": [
        "qMRI-qMRIGD",
        [
            "All_qMRI_T1",
            "All_qMRI_T2",
            "All_qMRI_PD",
            "All_qMRI_T1_GD",
            "All_qMRI_T2_GD",
            "All_qMRI_PD_GD",
        ],
    ],
    "6": [
        " cMRI-qMRI-qMRIGD",
        [
            "All_T1FLAIR_GD",
            "All_T1FLAIR",
            "All_T2FLAIR",
            "All_T2",
            "All_qMRI_T1",
            "All_qMRI_T2",
            "All_qMRI_PD",
            "All_T1FLAIR_GD",
            "All_T1FLAIR",
            "All_T2FLAIR",
            "All_T2",
            "All_qMRI_T1_GD",
            "All_qMRI_T2_GD",
            "All_qMRI_PD_GD",
        ],
    ],
    "7": ["T1WGD", ["All_T1FLAIR_GD"]],
    "8": [
        "Delta_Tumor_border",
        ["All_delta_border_R1_GD", "All_delta_border_R2_GD", "All_delta_border_PD_GD"],
    ],
    "9": [
        "cMRI-Delta_Tumor_border",
        [
            "All_T1FLAIR_GD",
            "All_T1FLAIR",
            "All_T2FLAIR",
            "All_T2",
            "All_delta_border_R1_GD",
            "All_delta_border_R2_GD",
            "All_delta_border_PD_GD",
        ],
    ],
    "10": ["T1wGD_T1GD_T2GD", ["All_T1FLAIR_GD", "All_qMRI_T1_GD", "All_qMRI_T2_GD"]],
    "11": ["qT1GD", ["All_qMRI_T1_GD"]],
    "12": ["qT2GD", ["All_qMRI_T2_GD"]],
    "13": ["qT1GD-qT2GD", ["All_qMRI_T2_GD"]],
    "14": ["T1WGD-qT1GD", ["All_T1FLAIR_GD", "All_qMRI_T1_GD"]],
    "15": ["qPDDG", ["All_qMRI_PD_GD"]],
    "16": ["qPD", ["All_qMRI_PD"]],
    "17": ["T1w-T1wGD-T2w", ["All_T1FLAIR_GD", "All_T1FLAIR", "All_T2"]],
    "18": ["T2w", ["All_T2"]],
    "19": ["FLAIR", ["All_T2FLAIR"]],
}


# --------------------------------------
# set GPU
# --------------------------------------

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args_dict["GPU_NBR"]

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

devices = tf.config.list_physical_devices("GPU")

if devices:
    print(f'Running training on GPU # {args_dict["GPU_NBR"]} \n')
    warnings.simplefilter(action="ignore", category=FutureWarning)
    tf.config.experimental.set_memory_growth(devices[0], True)
else:
    Warning(
        f"ATTENTION!!! MODEL RUNNING ON CPU. Check implementation in case GPU is wanted."
    )

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]


# %% 1.1 - open the train_val_test_subject.json file, get path to test subjects, and load data

with open(args_dict["TRAIN_VALIDATION_TEST_FILE"]) as file:
    config = json.load(file)

# build filename based on the model input specifications
test_gt_files = [
    os.path.join(args_dict["ANNOTATION_DATASET_FOLDER"], f) for f in config["test"]
]
# load annotation data
test_gt = []

ds = "test"
for idx, gt in enumerate(test_gt_files):
    print(f"Woring on {ds} annotation {idx+1}/{len(test_gt_files)}\r", end="")
    aus_annotations_archive = utilities.load_3D_data([gt])
    annotations = aus_annotations_archive["data_volumes"][:, :, :, :, 0]
    annotations = annotations.transpose(0, 3, 1, 2)
    annotations = annotations.reshape(
        annotations.shape[0] * annotations.shape[1],
        annotations.shape[2],
        annotations.shape[3],
    )
    annotations = annotations / np.max(annotations)
    test_gt.append(to_categorical(annotations.astype("int"), 2))

test_gt = np.concatenate(test_gt, axis=0)
test_gt_classification = (np.sum(test_gt[:, :, :, 1], axis=(1, 2)) > 0).astype(int)
test_gt_classification_logits = to_categorical(test_gt_classification, num_classes=2)

# load image data
input_configuration = os.path.basename(args_dict["MODEL_PATH"])
input_configuratio_idx = [
    k for k, value in combination_dict.items() if value[0] == input_configuration
][0]

modalities = combination_dict[input_configuratio_idx][1]

# check if all modaliites are available, if not print warning and move to the next setting
check_modalities = [
    m
    for m in modalities
    if os.path.isdir(os.path.join(args_dict["IMG_DATASET_FOLDER"], m)) == False
]
if check_modalities:
    print(
        f'\n {"#"*10} \n ATTENTION! \n Not all modalities are available {check_modalities}! Check dataset folder. \n Moving to the next setting. \n {"#"*10}'
    )
else:
    # create test, validation and training path for all the modalities
    file_dict = {"test": {}}
    for s in config[ds]:
        subject_number = s.split("_")[-1].split(".")[0]
        file_dict[ds][subject_number] = {}
        for m in modalities:
            file_dict[ds][subject_number][m] = os.path.join(
                args_dict["IMG_DATASET_FOLDER"],
                m,
                "_".join(m.split("_")[1::]) + "_" + s.split("_")[-1],
            )

# finally load image data data
test_img = []
# load test data
for s in file_dict[ds]:
    aus_subject = []
    for m in file_dict[ds][s]:
        print(
            f"Working on dataset {ds}, subject {s}, modality {m}          \r",
            end="",
        )
        aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
    test_img.append(np.concatenate(aus_subject, axis=3))
test_img = np.concatenate(test_img, axis=0)

print("\n")

# %% - LOOP THOURH ALL THE FOLDS AND TEST LAST AND BEST MODEL
last_best_model_names = {"last": "last_model", "best": "best_model_weights"}

summary_test_fold_case = {"last": [], "best": []}
summary_test_fold_logits = {"last": [], "best": []}
summary_test = {}

for f in range(args_dict["NBR_FOLDS"]):
    # load last model (for the best model, just load weights)
    model_path = os.path.join(
        args_dict["MODEL_PATH"], f"fold_{f+1}", last_best_model_names["last"]
    )
    # initialize where to save the test performance
    summary_test[str(f + 1)] = {"best": [], "last": []}
    for model_version in ["last", "best"]:
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            if model_version == "best":
                model_path = os.path.join(
                    args_dict["MODEL_PATH"],
                    f"fold_{f+1}",
                    last_best_model_names[model_version],
                    "",
                )
                # load weights best model
                try:
                    model.load_weights(model_path)
                except:
                    print(
                        f"Could not load best model weights for fold {f+1}. Given {model_path}"
                    )
            # if model is present, test it
            fold_pred_softmax = []
            for i in range(test_img.shape[0]):
                print(
                    f"Working on {model_version} model fold {f+1} ({i+1:3d}/{test_img.shape[0]})\r",
                    end="",
                )
                fold_pred_softmax.append(
                    model(np.expand_dims(test_img[i, :, :, :], axis=0)).numpy()[0]
                )
            # compute fold metrics
            summary_test[str(f + 1)][model_version] = utilities.get_performance_metrics(
                test_gt_classification_logits,
                np.array(fold_pred_softmax),
                average="macro",
            )
            # save additional information
            summary_test[str(f + 1)][model_version]["per_sample_pred"] = np.argmax(
                np.array(fold_pred_softmax), axis=1
            )

            summary_test[str(f + 1)][model_version][
                "per_sample_logits"
            ] = fold_pred_softmax

            summary_test[str(f + 1)][model_version][
                "per_sample_ground_truth"
            ] = test_gt_classification_logits

            # save fold logits
        except:
            print(f"Model not found for fold {f+1}. Given {model_path}")

print()
# %% 3 - PRINT SUMMARY TEST
metric_to_print = "matthews_correlation_coefficient"
print(f"METRIC : {metric_to_print}")
for i in range(len(summary_test)):
    print(
        f'Fold {i+1} -> last model:{summary_test[str(i+1)]["last"][metric_to_print]:0.3f}, best model:{summary_test[str(i+1)]["best"][metric_to_print]:0.3f}'
    )

for m in ["last", "best"]:
    aus = [
        summary_test[str(f + 1)][m][metric_to_print] for f in range(len(summary_test))
    ]
    print(
        f"{m} model (mean \u00B1 std) -> {np.mean(aus):0.3f} \u00B1 {np.std(aus):0.3f}"
    )

# %% SAVE INFORMATION FOR EACH FOLD SEPARATELY
for f in range(args_dict["NBR_FOLDS"]):
    summary_file = os.path.join(
        args_dict["MODEL_PATH"], f"fold_{str(f+1)}", f"tabular_test_summary.csv"
    )
    csv_file = open(summary_file, "w")
    writer = csv.writer(csv_file)
    csv_header = [
        "classification_type",
        "nbr_classes",
        "model_type",
        "model_version",
        "fold",
        "precision",
        "recall",
        "accuracy",
        "f1-score",
        "auc",
        "matthews_correlation_coefficient",
    ]
    writer.writerow(csv_header)
    # build rows to save in the csv file
    csv_rows = []
    for m in ["last", "best"]:
        csv_rows.append(
            [
                "tumor-vs-no_tumor",
                2,
                "Simple_2D_detection_CNN",
                m,
                f + 1,
                summary_test[str(f + 1)][m]["overall_precision"],
                summary_test[str(f + 1)][m]["overall_recall"],
                summary_test[str(f + 1)][m]["overall_accuracy"],
                summary_test[str(f + 1)][m]["overall_f1-score"],
                summary_test[str(f + 1)][m]["overall_auc"],
                summary_test[str(f + 1)][m]["matthews_correlation_coefficient"],
            ]
        )
    writer.writerows(csv_rows)
    csv_file.close()

# %% SAVE SUMMARY FOR ALL THE FOLDS IN ONE FILE
summary_file = os.path.join(args_dict["MODEL_PATH"], f"tabular_test_summary.csv")
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
csv_header = [
    "classification_type",
    "nbr_classes",
    "model_type",
    "model_version",
    "fold",
    "precision",
    "recall",
    "accuracy",
    "f1-score",
    "auc",
    "matthews_correlation_coefficient",
]
writer.writerow(csv_header)
# build rows to save in the csv file
csv_rows = []
for f in range(args_dict["NBR_FOLDS"]):
    for m in ["last", "best"]:
        csv_rows.append(
            [
                "tumor-vs-no_tumor",
                2,
                "Simple_2D_detection_CNN",
                m,
                f + 1,
                summary_test[str(f + 1)][m]["overall_precision"],
                summary_test[str(f + 1)][m]["overall_recall"],
                summary_test[str(f + 1)][m]["overall_accuracy"],
                summary_test[str(f + 1)][m]["overall_f1-score"],
                summary_test[str(f + 1)][m]["overall_auc"],
                summary_test[str(f + 1)][m]["matthews_correlation_coefficient"],
            ]
        )
writer.writerows(csv_rows)
csv_file.close()

# %% SAVE DICT WITH LOGITS
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


summary_file = os.path.join(args_dict["MODEL_PATH"], f"summary_test_logits.json")
with open(summary_file, "w") as fp:
    json.dump(summary_test, fp, cls=NpEncoder)

# # %% TEST OPEN FILE
# with open(summary_file) as file:
#     test = json.load(file)
