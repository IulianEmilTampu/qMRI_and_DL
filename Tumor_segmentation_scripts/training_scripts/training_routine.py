# %%

"""
Main script that runs model training for tumor segmentation (tumor core) in the context of the qMRI project.

Steps
1 - import data and annotations based on the input configuration setting
3 - train a classifier (2D UNet) for the segmentation of the tumor in each slide
4 - save model
"""
import os

import glob
import json
import csv
import warnings
import numpy as np
import argparse
import importlib
import logging
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# local imports
import utilities
import models

#%% GET VARIABLES
to_print = "    RUN TRAINING ON MULTIPLE DATA CONFIGURATIONS   "

print(f'\n{"-"*len(to_print)}')
print(to_print)
print(f'{"-"*len(to_print)}\n')

su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Run cross validation training on a combination of MRI modalities."
    )
    parser.add_argument(
        "-wf",
        "--WORKING_FOLDER",
        required=True,
        type=str,
        help="Provide the working folder where the trained model will be saved.",
    )
    parser.add_argument(
        "-df",
        "--IMG_DATASET_FOLDER",
        required=True,
        type=str,
        help="Provide the Image Dataset Folder where the folders for each modality are located (see dataset specifications in the README file).",
    )
    parser.add_argument(
        "-af",
        "--ANNOTATION_DATASET_FOLDER",
        required=True,
        type=str,
        help="Provide the Annotation Folder where annotations are located (see dataset specifications in the README file).",
    )
    parser.add_argument(
        "-bmf",
        "--BRAIN_MASK_FOLDER",
        required=True,
        type=str,
        help="Provide the Brain Mask Folder where annotations are located (see dataset specifications in the README file).",
    )
    parser.add_argument(
        "-gpu",
        "--GPU_NBR",
        default=0,
        type=str,
        help="Provide the GPU number to use for training.",
    )
    parser.add_argument(
        "-model_name",
        "--MODEL_NAME",
        required=False,
        type=str,
        default="2D_UNet_segmentation_model",
        help="Name used to save the model and the scores",
    )
    parser.add_argument(
        "-ns",
        "--NBR_SUBJECTS",
        type=int,
        default=21,
        help="Number of subjects to use during training.",
    )
    parser.add_argument(
        "-n_folds",
        "--NBR_FOLDS",
        required=False,
        type=int,
        default=1,
        help="Number of cross validation folds.",
    )
    parser.add_argument(
        "-lr",
        "--LEARNING_RATE",
        required=False,
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "-batch_size",
        "--BATCH_SIZE",
        required=False,
        type=int,
        default=16,
        help="Specify batch size. Default 16",
    )
    parser.add_argument(
        "-e",
        "--MAX_EPOCHS",
        required=False,
        type=int,
        default=300,
        help="Number of max training epochs.",
    )
    parser.add_argument(
        "-dc",
        "--DATASET_CONFIGURATION",
        required=False,
        nargs="+",
        default=0,
        help="Which dataset configuration to train the model on",
    )

    # tumor border as extra channel parameters
    parser.add_argument(
        "-add_tumor_border",
        "--ADD_TUMOR_BORDER",
        required=False,
        type=bool,
        default=False,
        help="If True, along with the tumor and background, the tumor border will be added to the ground truth. The size of the tumor border is defined by the parameter -tumor_border_width.",
    )
    parser.add_argument(
        "-tumor_border_width",
        "--TUMOR_BORDER_WIDTH",
        required=False,
        type=int,
        default=4,
        help="Defines the width of the tumor border in pixels. Used in case -add_tumor_border is True",
    )

    # tumor erosion or expansion parameters
    parser.add_argument(
        "-adjust_tumor",
        "--ADJUST_TUMOR_SIZE",
        required=False,
        default=None,
        help="Used to specify if the tumor mask should be expanded or eroded (expand OR erode). The amount of expansion of erosion is specified by the parapeter -n_pixels.",
    )
    parser.add_argument(
        "-n_pixels",
        "--NBR_PIXELS_TUMOR_ADJUSTMENT",
        required=False,
        type=int,
        default=4,
        help="Defines the amount of erosion or expansion of the tumor mask. Used if -adjust_tumor is set to expand or erode",
    )

    # other parameters
    parser.add_argument(
        "-rns",
        "--RANDOM_SEED_NUMBER",
        required=False,
        type=int,
        default=29122009,
        help="Specify random number seed. Useful to have models trained and tested on the same data.",
    )

    args_dict = dict(vars(parser.parse_args()))
    # bring variable to the right format
    args_dict["DATASET_CONFIGURATION"] = [
        int(a) for a in args_dict["DATASET_CONFIGURATION"]
    ]

else:
    # # # # # # # # # # # # # # DEBUG
    args_dict = {
        "WORKING_FOLDER": "/flush/iulta54/Research/qMRI_and_DL/Tumor_segmentation_scripts/training_scripts",
        "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality",
        "ANNOTATION_DATASET_FOLDER": "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality/All_Annotations",
        "BRAIN_MASK_FOLDER": "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality/All_Brain_masks",
        "GPU_NBR": "0",
        "MODEL_NAME": "Test",
        "NBR_SUBJECTS": 5,
        "NBR_FOLDS": 2,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 2,
        "MAX_EPOCHS": 10,
        "DATASET_CONFIGURATION": [5],
        # "DATASET_CONFIGURATION": [0,1,2,7,10],
        # "DATASET_CONFIGURATION": [11,12,13,14,15,16],
        # "DATASET_CONFIGURATION": [0, 19],
        "ADD_TUMOR_BORDER": False,
        "TUMOR_BORDER_WIDTH": None,
        "ADJUST_TUMOR_SIZE": False,
        "NBR_PIXELS_TUMOR_ADJUSTMENT": None,
        "RANDOM_SEED_NUMBER": 29122009,
    }

# --------------------------------------
# set GPU
# --------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args_dict["GPU_NBR"]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import tensorflow
import tensorflow as tf
import warnings

tf.get_logger().setLevel(logging.ERROR)
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

# -------------------------------------
# Check that the given folder exist
# -------------------------------------
for folder, fd in zip(
    [
        args_dict["WORKING_FOLDER"],
        args_dict["IMG_DATASET_FOLDER"],
        args_dict["ANNOTATION_DATASET_FOLDER"],
        args_dict["BRAIN_MASK_FOLDER"],
    ],
    [
        "working folder",
        "image dataset folder",
        "annotation folder",
        "brain mask folder",
    ],
):
    if not os.path.isdir(folder):
        raise ValueError(f"{fd.capitalize} not found. Given {folder}.")

# -------------------------------------
# Create folder where to save the model
# -------------------------------------
args_dict["SAVE_PATH"] = os.path.join(
    args_dict["WORKING_FOLDER"], "trained_models_archive", args_dict["MODEL_NAME"]
)
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)


if not su_debug_flag:
    # save training configuration
    with open(os.path.join(args_dict["SAVE_PATH"], "config.json"), "w") as config_file:
        config_file.write(json.dumps(args_dict))

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %% SPECIFY INPUT DATA COMBINATION DICTIONARY
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

# ----------------------------------
# Set train test and validation sets
# ----------------------------------
print(f"Splitting dataset (per-volume splitting).")
"""
Independently from the combination of modalities, the test valid and train sets
defined so that no vlomume is present in more than one set.

Steps
2 - using the number of subject to use, create indexes to identify which files
    are used for training, validation and test
3 - save the infromation about the split.
"""


subj_index = [i for i in range(args_dict["NBR_SUBJECTS"])]
subj_train_val_idx, subj_test_idx = train_test_split(
    subj_index, test_size=0.1, random_state=args_dict["RANDOM_SEED_NUMBER"]
)
print(f'{"# Train-val subjects":18s}: {len(subj_train_val_idx):2d}')
print(f'{"# Test subjects":18s}: {len(subj_test_idx):2d} ({subj_test_idx})')
subj_train_idx = []
subj_val_idx = []
# set cross validation
if args_dict["NBR_FOLDS"] > 1:
    kf = KFold(
        n_splits=args_dict["NBR_FOLDS"],
        shuffle=True,
        random_state=args_dict["RANDOM_SEED_NUMBER"],
    )
    for idx, (train_index, val_index) in enumerate(kf.split(subj_train_val_idx)):
        subj_train_idx.append([subj_train_val_idx[i] for i in train_index])
        subj_val_idx.append([subj_train_val_idx[i] for i in val_index])

        # print to check that all is good
        print(
            f'Fold {idx+1}: \n {""*4}{"training":10s} ->{subj_train_idx[-1]} \n {""*4}{"validation":10s} ->{subj_val_idx[-1]}'
        )
else:
    # args_dict['NBR_FOLDS'] is one, setting 10% of the training dataset as validation
    aus_train, aus_val = train_test_split(
        subj_train_val_idx, test_size=0.1, random_state=args_dict["RANDOM_SEED_NUMBER"]
    )
    subj_train_idx.append(aus_train)
    subj_val_idx.append(aus_val)

    # print to check that all is good
    print(
        f'Fold {args_dict["NBR_FOLDS"]}: \n {""*4}{"training":10s} ->{subj_train_idx[-1]} \n {""*4}{"validation":10s} ->{subj_val_idx[-1]}'
    )

# -------------------------------
# Load and preprocess annotations
# -------------------------------
print(f"\nLoading and preprocessing annotations...")

if any(
    [
        args_dict["ADJUST_TUMOR_SIZE"] == "expand",
        args_dict["ADJUST_TUMOR_SIZE"] == "erode",
    ]
):
    print(
        f'Adjusting tumor by {args_dict["ADJUST_TUMOR_SIZE"]} ({args_dict["ADJUST_TUMOR_SIZE"]} set for {args_dict["NBR_PIXELS_TUMOR_ADJUSTMENT"]} pixels)'
    )

if args_dict["ADD_TUMOR_BORDER"]:
    print(
        f'Adding tumor border to the ground truth (setting border to {args_dict["TUMOR_BORDER_WIDTH"]} pixels)'
    )

classes = ["Background", "Tumor"]
Nclasses = len(classes)

annotation_file_names = glob.glob(
    os.path.join(args_dict["ANNOTATION_DATASET_FOLDER"], "*.nii.gz")
)
annotation_file_names.sort()
annotation_file_names = annotation_file_names[0 : args_dict["NBR_SUBJECTS"]]

brain_mask_file_names = glob.glob(
    os.path.join(args_dict["BRAIN_MASK_FOLDER"], "*.nii.gz")
)
brain_mask_file_names.sort()
brain_mask_file_names = brain_mask_file_names[0 : args_dict["NBR_SUBJECTS"]]

# laod all files and save in a list since new need to able to easily index them
annotations_archive = []
brain_mask_archive = []
for idx, (a_file, bm_file) in enumerate(
    zip(annotation_file_names, brain_mask_file_names)
):
    print(f'{" "*2} Subject: {idx+1}/{len(annotation_file_names)}\r', end="")
    aus_annotations_archive = utilities.load_3D_data([a_file])
    annotations = aus_annotations_archive["data_volumes"][:, :, :, :, 0]
    annotations = annotations.transpose(0, 3, 1, 2)
    annotations = annotations.reshape(
        annotations.shape[0] * annotations.shape[1],
        annotations.shape[2],
        annotations.shape[3],
    )
    annotations = annotations / np.max(annotations)
    annotations = annotations.astype("int")

    # load brain masks as well
    aus_brain_mask_archive = utilities.load_3D_data([bm_file])
    brain_mask = aus_brain_mask_archive["data_volumes"][:, :, :, :, 0]
    brain_mask = brain_mask.transpose(0, 3, 1, 2)
    brain_mask = brain_mask.reshape(
        brain_mask.shape[0] * brain_mask.shape[1],
        brain_mask.shape[2],
        brain_mask.shape[3],
    )
    brain_mask = brain_mask / np.max(brain_mask)
    brain_mask_archive.append(brain_mask.astype("int"))

    # expand or erode tumor mask if needed
    if any(
        [
            args_dict["ADJUST_TUMOR_SIZE"] == "expand",
            args_dict["ADJUST_TUMOR_SIZE"] == "erode",
        ]
    ):
        for i in range(annotations.shape[0]):
            # print('before', annotations[i,:,:].sum())
            annotations[i, :, :] = utilities.args_dict["ADJUST_TUMOR_SIZE"](
                np.squeeze(annotations[i, :, :]),
                args_dict["NBR_PIXELS_TUMOR_ADJUSTMENT"]
                if args_dict["ADJUST_TUMOR_SIZE"] == "expand"
                else -args_dict["NBR_PIXELS_TUMOR_ADJUSTMENT"],
            )
            # print('after', annotations[i,:,:].sum() )

    # add tumor border to the annotation in case needed
    if args_dict["ADD_TUMOR_BORDER"]:
        for i in range(annotations.shape[0]):
            border = (
                utilities.get_tumor_border(
                    np.squeeze(annotations[i, :, :]), args_dict["TUMOR_BORDER_WIDTH"]
                )
                * 2
            )
            annotations[i, :, :] += border

        # set other parameters
        classes = ["Background", "Tumor", "Tumor_border"]
        Nclasses = len(classes)

    # fix annotation overflow if annotation was expanded or tumor border was added
    if any(
        [args_dict["ADJUST_TUMOR_SIZE"] == "expand", args_dict["ADJUST_TUMOR_SIZE"]]
    ):
        for i in range(annotations.shape[0]):
            annotations[i, :, :] = (
                annotations[i, :, :]
                * brain_mask[
                    i,
                    :,
                    :,
                ]
            )

    annotations_archive.append(to_categorical(annotations, Nclasses))

# take out test annotations
Ytest = [annotations_archive[i] for i in subj_test_idx]
Ytest = np.concatenate(Ytest, axis=0)

print("TEST - Max mask value after processing is ", np.max(Ytest))
print("TEST - Min mask value after processing is ", np.min(Ytest))
print("TEST - Unique mask values after processing are ", np.unique(Ytest))

print(f"TEST - {'Total number of tumour voxels is':37s}: {np.sum(Ytest[:,:,:,1])}")
print(f"TEST - {'Total number of background voxels is':37s}: {np.sum(Ytest[:,:,:,0])}")
print(
    f"TEST - {'Proportion of tumour voxels is':37s}: {np.sum(Ytest[:,:,:,1]) / np.sum(Ytest[:,:,:,0])}"
)


# %% Visualize annotations
# importlib.reload(utilities)
# utilities.inspectDataset_v2([annotations_archive[0], np.expand_dims(brain_mask_archive[0],-1)], start_slice=0, end_slice=24)

# %% Save infromation about which subjects are used for training/validation/testing

dict = {
    "test": [os.path.basename(annotation_file_names[i]) for i in subj_test_idx],
    "train": [],
    "validation": [],
}

# build indexes of subjects to use for training and validation during the cross-valudation routine
for f in range(args_dict["NBR_FOLDS"]):
    dict["train"].append(
        [os.path.basename(annotation_file_names[i]) for i in subj_train_idx[f]]
    )
    dict["validation"].append(
        [os.path.basename(annotation_file_names[i]) for i in subj_val_idx[f]]
    )

# save cross validation subject information
with open(
    os.path.join(args_dict["SAVE_PATH"], "train_val_test_subjects.json"), "w"
) as file:
    json.dump(dict, file)

# save training information
with open(os.path.join(args_dict["SAVE_PATH"], "config.json"), "w") as file:
    json.dump(args_dict, file)

# %% Training with different combinations of input data

diceScores = np.zeros((len(combination_dict), args_dict["NBR_FOLDS"]))

for dataCombination in args_dict["DATASET_CONFIGURATION"]:
    # load all the subjects and respective modalities specified by the
    # combination dictionary. Here we want to save each subject independetly
    # so that it is easier later to select the subject for the train/val/test
    # split
    image_archive = []

    setting = combination_dict[str(dataCombination)][0]
    modalities = combination_dict[str(dataCombination)][1]

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
        # create folder for the combination
        if not os.path.isdir(os.path.join(args_dict["SAVE_PATH"], setting)):
            os.mkdir(os.path.join(args_dict["SAVE_PATH"], setting))

        print(f"\nTraining on {setting} setting. Loading files...")
        for subject in range(args_dict["NBR_SUBJECTS"]):
            aus_images = []
            # load all modalities
            for idx, modality in enumerate(modalities):
                print(
                    f'{" "*2}Subject {subject+1:2d}/{args_dict["NBR_SUBJECTS"]:2d} -> {idx+1:2d}/{len(modalities):2d} {modality}    \r',
                    end="",
                )
                # get file paths and sort
                aus_img_files = glob.glob(
                    os.path.join(args_dict["IMG_DATASET_FOLDER"], modality, "*.nii.gz")
                )
                aus_img_files.sort()
                aus_img_files = aus_img_files[0 : args_dict["NBR_SUBJECTS"]]
                # load data
                aus_images.append(utilities.load_MR_modality([aus_img_files[subject]]))

            image_archive.append(np.concatenate(aus_images, axis=3))

        print(f'{" "*3}Setting test dataset')
        Xtest = [image_archive[i] for i in subj_test_idx]
        Xtest = np.concatenate(Xtest, axis=0)

        # creating fold folders where to save the trained models
        for f in range(args_dict["NBR_FOLDS"]):
            if not os.path.isdir(
                os.path.join(args_dict["SAVE_PATH"], setting, "fold_" + str(f + 1))
            ):
                os.mkdir(
                    os.path.join(args_dict["SAVE_PATH"], setting, "fold_" + str(f + 1))
                )

        # create dictionary where to save the test performance for this cross validation
        summary_test = dict.fromkeys(range(args_dict["NBR_FOLDS"]))
        for key in summary_test.keys():
            summary_test[key] = {
                "best": {"Dice": 0, "Accuracy": 0},
                "last": {"Dice": 0, "Accuracy": 0},
            }

        # RUNIING CROSS VALIDATION
        for cv_f in range(args_dict["NBR_FOLDS"]):
            print(f'{" "*3}Setting up training an validation sets...')

            # images
            Xtrain = [image_archive[i] for i in subj_train_idx[cv_f]]
            Xtrain = np.concatenate(Xtrain, axis=0)

            Xvalid = [image_archive[i] for i in subj_val_idx[cv_f]]
            Xvalid = np.concatenate(Xvalid, axis=0)

            # annotations (use brain mask to set to zero eventual annotations exxpanded outside the brain)
            Ytrain = [annotations_archive[i] for i in subj_train_idx[cv_f]]
            Ytrain = np.concatenate(Ytrain, axis=0)

            Yvalid = [annotations_archive[i] for i in subj_val_idx[cv_f]]
            Yvalid = np.concatenate(Yvalid, axis=0)

            # shuffle training dataset
            print(f'{" "*3}Shuffling training dataset...')
            Xtrain, Ytrain = utilities.shuffle_array(Xtrain, Ytrain)

            # print datasets shapes
            print(
                f'{" "*3}{"Training dataset shape":25s}: image -> {Xtrain.shape}, mask -> {Ytrain.shape}'
            )
            print(
                f'{" "*3}{"Validation dataset shape":25s}: image -> {Xvalid.shape}, mask -> {Yvalid.shape}'
            )
            print(
                f'{" "*3}{"Testing dataset shape":25s}: image -> {Xtest.shape}, mask -> {Ytest.shape}'
            )

            # # DEBUG
            # # check files
            #
            # x = Xtest
            # y = Ytest
            # utilities.inspectDataset(Xtrain, Ytrain, end_slice=100)
            # # end check files

            # --------------------------
            # CREATE DATA GENERATORS
            # -------------------------

            train_gen = utilities.create_data_gen(
                Xtrain,
                Ytrain,
                batch_size=args_dict["BATCH_SIZE"],
                seed=args_dict["RANDOM_SEED_NUMBER"],
            )
            val_gen = utilities.create_data_gen(
                Xvalid,
                Yvalid,
                batch_size=args_dict["BATCH_SIZE"],
                seed=args_dict["RANDOM_SEED_NUMBER"],
            )

            # ## Check generator

            # x = []
            # y = []
            # for idx, (i,j) in enumerate(val_gen):
            #     x.append(i)
            #     y.append(j)
            #
            #     if idx == 10:
            #         break
            #
            #
            # x = np.concatenate(x, axis=0)
            # y = np.concatenate(y, axis=0)
            #
            # importlib.reload(utilities)
            # utilities.plotEpochSegmentation(x,
            #                         y,
            #                         y,
            #                         save_path='/home/iulta54/Code/P4-qMRI/trained_models/Simple_CNN_rkv_10_fold5_lr0.01_batch2_cv_repetition_1_seed_1234/test',
            #                         epoch=1,
            #                         display=False,
            #                         n_subjects_to_show=3)
            #
            # utilities.inspectDataset(x, y, end_slice=x.shape[0])
            #
            # ## end check generator

            # --------------------------
            # BUILD MODEL
            # -------------------------

            # Always use the same class weight
            if args_dict["ADD_TUMOR_BORDER"]:
                class_weights = [1, 60000, 60000]
            else:
                class_weights = [
                    1,
                    int((np.sum(Ytrain[:, :, :, 0]) / np.sum(Ytrain[:, :, :, 1])) ** 2),
                ]

            img_size = Xtrain[0].shape
            net_aug = models.unet(
                img_size,
                Nclasses,
                class_weights,
                args_dict["MODEL_NAME"],
                Nfilter_start=32,
                batch_size=args_dict["BATCH_SIZE"],
                depth=4,
            )

            # --------------------------
            # TRAIN MODEL
            # -------------------------

            net_aug.custum_train(
                train_gen,
                val_gen,
                Xtrain.shape[0] / args_dict["BATCH_SIZE"],
                Xvalid.shape[0] / args_dict["BATCH_SIZE"],
                max_epocs=args_dict["MAX_EPOCHS"],
                verbose=1,
                save_model_path=os.path.join(
                    args_dict["SAVE_PATH"], setting, "fold_" + str(cv_f + 1)
                ),
                early_stopping=True,
                patience=args_dict["MAX_EPOCHS"],
                start_learning_rate=args_dict["LEARNING_RATE"],
            )

            # --------------------------
            # EVALUATE LAST & BEST MODEL
            # -------------------------
            # last model
            acc, dice = net_aug.evaluate(Xtest, Ytest)
            # diceScores[dataCombination,cv_f] = dice
            summary_test[cv_f]["last"]["Dice"] = dice
            summary_test[cv_f]["last"]["Accuracy"] = acc

            # best model
            # load weights
            net_aug.model.load_weights(
                os.path.join(
                    args_dict["SAVE_PATH"],
                    setting,
                    "fold_" + str(cv_f + 1),
                    "best_model_weights.tf",
                )
            )
            acc, dice = net_aug.evaluate(Xtest, Ytest)
            # diceScores[dataCombination,cv_f] = dice
            summary_test[cv_f]["best"]["Dice"] = dice
            summary_test[cv_f]["best"]["Accuracy"] = acc

            # -------------------------------------
            # SAVE MODEL PORFORMANCE FOR THIS FOLD
            # -------------------------------------

            # save information for this fold
            summary_file = os.path.join(
                args_dict["SAVE_PATH"],
                setting,
                f"fold_{str(cv_f+1)}",
                f"tabular_test_summary.csv",
            )
            csv_file = open(summary_file, "w")
            writer = csv.writer(csv_file)
            csv_header = [
                "task",
                "nbr_classes",
                "input_configuration",
                "model_type",
                "model_version",
                "fold",
                "Dice",
                "Accuracy",
            ]
            writer.writerow(csv_header)
            # build rows to save in the csv file
            csv_rows = []
            for m in ["last", "best"]:
                csv_rows.append(
                    [
                        "segmentation",
                        2,
                        setting,
                        "2D_UNet",
                        m,
                        cv_f + 1,
                        summary_test[cv_f][m]["Dice"],
                        summary_test[cv_f][m]["Accuracy"],
                    ]
                )
            writer.writerows(csv_rows)
            csv_file.close()

    # ----------------------------------------------------------------------
    # SAVE MODEL PORFORMANCE FOR ALL THE FOLDS FOR THIS INPUT CONFIGURATION
    # ----------------------------------------------------------------------

    summary_file = os.path.join(
        args_dict["SAVE_PATH"], setting, f"tabular_test_summary.csv"
    )
    csv_file = open(summary_file, "w")
    writer = csv.writer(csv_file)
    csv_header = [
        "task",
        "nbr_classes",
        "input_configuration",
        "model_type",
        "model_version",
        "fold",
        "Dice",
        "Accuracy",
    ]
    writer.writerow(csv_header)
    # build rows to save in the csv file
    csv_rows = []
    for cv_f in range(args_dict["NBR_FOLDS"]):
        for m in ["last", "best"]:
            csv_rows.append(
                [
                    "segmentation",
                    2,
                    setting,
                    "2D_UNet",
                    m,
                    cv_f + 1,
                    summary_test[cv_f][m]["Dice"],
                    summary_test[cv_f][m]["Accuracy"],
                ]
            )
    writer.writerows(csv_rows)
    csv_file.close()
