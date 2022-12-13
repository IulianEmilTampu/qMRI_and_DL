# %%
"""
Main script that runs model training for tumor detection (binary classification of if a 2D transversal image contains or not tumor) 
in the context of the qMRI project.

Steps
1 - import data and annotations
2 - based on the annotations, create labels for each slice
    slice without tumor = 0
    slice with tumor = 1
3 - train a classifier (simple 2D CNN) for the classification of the slices
4 - save model
"""
import os
import glob
import csv
import json
import warnings
import numpy as np
import argparse
import importlib
import logging
from pathlib import Path

from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# local imports
import utilities
import detection_models

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
        default="myModel",
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
        "WORKING_FOLDER": "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/training_scripts",
        "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality",
        "ANNOTATION_DATASET_FOLDER": "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality/All_Annotations",
        "BRAIN_MASK_FOLDER": "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality/All_Brain_masks",
        "GPU_NBR": "0",
        "MODEL_NAME": "Test",
        "NBR_SUBJECTS": 21,
        "NBR_FOLDS": 3,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 16,
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
from tensorflow_addons.optimizers import Lookahead

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

# %% Set train test and validation sets
print(f"Splitting dataset (per-volume splitting).")
"""
Independently from the combination of modalities, the test validation and train sets
are defined so that no vlomume is present in more than one set.

Steps
2 - using the number of subject to use, create indexes to identify which files
    are used for training, validation and testing
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
    # N_FOLDS is only one, setting 10% of the training dataset as validation
    aus_train, aus_val = train_test_split(
        subj_train_val_idx, test_size=0.1, random_state=args_dict["RANDOM_SEED_NUMBER"]
    )
    subj_train_idx.append(aus_train)
    subj_val_idx.append(aus_val)

    # print to check that all is good
    print(
        f'Fold {args_dict["NBR_FOLDS"]}: \n {""*4}{"training":10s} ->{subj_train_idx[-1]} \n {""*4}{"validation":10s} ->{subj_val_idx[-1]}'
    )

# %% Load and preprocess annotations
print(f"\nLoading and preprocessing annotations...")

if any(
    [
        args_dict["ADJUST_TUMOR_SIZE"] == "expand",
        args_dict["ADJUST_TUMOR_SIZE"] == "erode",
    ]
):
    print(
        f"Adjusting tumor ({args_dict['ADJUST_TUMOR_SIZE']} set for {args_dict['NBR_PIXELS_TUMOR_ADJUSTMENT']} pixels)"
    )

if args_dict["ADD_TUMOR_BORDER"]:
    print(
        f"Adding tumor border to the ground truth (setting border to {args_dict['TUMOR_BORDER_WIDTH']} pixels)"
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
            annotations[i, :, :] = utilities.adjust_tumor(
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
    if any([args_dict["ADJUST_TUMOR_SIZE"] == "expand", args_dict["ADD_TUMOR_BORDER"]]):
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
Ytest_classification = (np.sum(Ytest[:, :, :, 1], axis=(1, 2)) > 0).astype(int)

print("TEST - Max mask value after processing is ", np.max(Ytest))
print("TEST - Min mask value after processing is ", np.min(Ytest))
print("TEST - Unique mask values after processing are ", np.unique(Ytest))

print(f"TEST - {'Total number of tumour voxels is':37s}: {np.sum(Ytest[:,:,:,1])}")
print(f"TEST - {'Total number of background voxels is':37s}: {np.sum(Ytest[:,:,:,0])}")
print(
    f"TEST - {'Proportion of tumour voxels is':37s}: {np.sum(Ytest[:,:,:,1]) / np.sum(Ytest[:,:,:,0])}"
)

# print which subjects are used for testing in each run)
print(
    f"Testing on {[os.path.basename(annotation_file_names[i]) for i in subj_test_idx]}"
)

# #################### DEBUG - visualize annotations

# utilities.inspectDataset_v2([annotations_archive[0], np.expand_dims(brain_mask_archive[0],-1)], start_slice=0, end_slice=24)

# #################### DEBUG - end

# Save infromation about which subjects are used for training/validation/testing

dict = {
    "test": [os.path.basename(annotation_file_names[i]) for i in subj_test_idx],
    "train": [],
    "validation": [],
}

for f in range(args_dict["NBR_FOLDS"]):
    dict["train"].append(
        [os.path.basename(annotation_file_names[i]) for i in subj_train_idx[f]]
    )
    dict["validation"].append(
        [os.path.basename(annotation_file_names[i]) for i in subj_val_idx[f]]
    )

with open(
    os.path.join(args_dict["SAVE_PATH"], "train_val_test_subjects.json"), "w"
) as file:
    json.dump(dict, file)

# ---------------------------------
# Training with different combinations of input data
# ---------------------------------

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
        model_save_path = os.path.join(args_dict["SAVE_PATH"], setting)
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

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

        # create dictionary where to save the test performance
        summary_test = {}

        # RUNIING CROSS VALIDATION
        for cv_f in range(args_dict["NBR_FOLDS"]):
            # make forder where to save the model
            save_model_path = os.path.join(
                args_dict["SAVE_PATH"], setting, f"fold_{cv_f+1}"
            )
            Path(save_model_path).mkdir(parents=True, exist_ok=True)
            summary_test[str(cv_f + 1)] = {"best": [], "last": []}

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

            # --------------------------
            # CREATE DATA GENERATORS
            # -------------------------
            importlib.reload(utilities)

            # Setup generators for augmentation
            importlib.reload(utilities)

            train_gen = utilities.create_data_gen(
                Xtrain,
                Ytrain,
                batch_size=args_dict["BATCH_SIZE"],
                seed=args_dict["RANDOM_SEED_NUMBER"],
                classification=True,
            )
            val_gen = utilities.create_data_gen(
                Xvalid,
                Yvalid,
                batch_size=args_dict["BATCH_SIZE"],
                seed=args_dict["RANDOM_SEED_NUMBER"],
                classification=True,
            )
            test_gen = utilities.create_data_gen(
                Xtest,
                Ytest,
                batch_size=1,
                seed=args_dict["RANDOM_SEED_NUMBER"],
                classification=True,
                test_set=True,
            )

            ## LOAD CLASSIFICATION MODEL
            # set up flags (HERE ONE CAN SPECIFY IF TO USE OFF-THE-SHELF MODELS SUCH AS ResNet or VGG.
            # The weight of these models can be loaded using Keras or from a directory. Here is an example for when the models
            # are stored into a directory in the working folder (Models_library))

            model_version = "Simple_CNN"
            model_weights = "random"

            dict_models = {
                "ResNet50": {
                    "random": os.path.join(
                        args_dict["WORKING_FOLDER"],
                        "Models_library",
                        "tf21_ResNet50_512",
                    ),
                    "pre_trained": os.path.join(
                        args_dict["WORKING_FOLDER"],
                        "Models_library",
                        "tf21_ResNet50_224_pre_trained",
                    ),
                },
                "MobileNet_v2": {
                    "random": os.path.join(
                        args_dict["WORKING_FOLDER"],
                        "Models_library",
                        "tf21_MobileNet_v2_512",
                    ),
                    "pre_trained": os.path.join(
                        args_dict["WORKING_FOLDER"],
                        "Models_library",
                        "tf21_MobileNet_v2_224_pre_trained",
                    ),
                },
            }

            # load the right model
            if any([k == model_version for k in dict_models.keys()]):
                encoder = tf.keras.models.load_model(
                    dict_models[model_version][model_weights]
                )
                nbr_input_channels_encoder = encoder.input_shape[-1]
                if model_weights == "pre_trained":
                    encoder.trainable = False

            print(
                f"Model {model_version} with {model_weights} weights. Building model..."
            )

            """
            BUILD USING SEQUENTIAL IF NOT THE MODEL WILL NOT LOAD AFTERWARDS!!! NOTE ALSO THAT EVEN IF THE MODEL HAS AN INPUT LAYER, 
            IT WILL BE CREATED WITHOUT ONE, SO WHEN LOADED, THE MODEL NEEDS TO BE BUILD AND COMPILED TO HAVE IT RUNNING.
            """

            # fix input size based on pre_trained model requirements
            if model_weights == "pre_trained":
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.InputLayer(input_shape=Xtrain[0].shape))

                # crop image to the right size
                encoder_input_shape = encoder._feed_input_shapes[0][1:3]
                crop_width = int((Xtrain.shape[1] - encoder_input_shape[0]) / 2)
                crop_hight = int((Xtrain.shape[2] - encoder_input_shape[1]) / 2)

                model.add(tf.keras.layers.Cropping2D(cropping=(crop_width, crop_hight)))

                # bring the images to have the right number of channels
                model.add(
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=3, activation="relu", padding="same"
                    )
                )
                model.add(
                    tf.keras.layers.Conv2D(
                        filters=nbr_input_channels_encoder,
                        kernel_size=3,
                        activation="relu",
                        padding="same",
                    )
                )

                model.add(encoder)
                # average pool the output of the encoder
                model.add(tf.keras.layers.GlobalAveragePooling2D())
                model.add(tf.keras.layers.Dropout(0.2))

                # and through the classifier
                model.add(tf.keras.layers.Dense(units=128, activation="relu"))
                model.add(tf.keras.layers.Dense(units=2, activation="softmax"))

            else:
                # build custom model (WHAT HAS BEEN USED IN THE qMRI PROJECT)
                model = detection_models.SimpleDetectionModel(
                    num_classes=2,
                    input_shape=Xtrain.shape[1::],
                    class_weights=None,
                    kernel_size=(3, 3),
                    pool_size=(2, 2),
                    model_name="SimpleDetectionModel",
                )

            ## COMPILE MODEL
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args_dict["LEARNING_RATE"]
            )
            optimizer = Lookahead(optimizer, sync_period=5, slow_step_size=0.5)

            loss = tf.keras.losses.BinaryCrossentropy()

            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

            ## SET MODEL CHECKPOINT
            best_model_path = os.path.join(save_model_path, "best_model_weights", "")
            Path(best_model_path).mkdir(parents=True, exist_ok=True)

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                save_weights_only=True,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            )

            ## RUN MODEL TRAINING
            history = model.fit(
                train_gen,
                steps_per_epoch=Xtrain.shape[0] / args_dict["BATCH_SIZE"],
                shuffle=True,
                validation_data=val_gen,
                validation_steps=Xvalid.shape[0] / args_dict["BATCH_SIZE"],
                epochs=args_dict["MAX_EPOCHS"],
                verbose=1,
                callbacks=[model_checkpoint_callback],
            )
            # save last model
            model.save(os.path.join(save_model_path, "last_model"))

            ## EVALUATE LAST & BEST MODEL
            importlib.reload(utilities)
            # ###################### LAST MODEL
            # get the per_slice classification
            Ptest_softmax = []
            for i in range(Xtest.shape[0]):
                Ptest_softmax.append(
                    model(np.expand_dims(Xtest[i, :, :, :], axis=0)).numpy()[0]
                )
            Ptest_softmax = np.array(Ptest_softmax)
            Ptest = np.argmax(Ptest_softmax, axis=-1)

            Ytest_logits = to_categorical(Ytest_classification, num_classes=2)
            summary_test[str(cv_f + 1)]["last"] = utilities.get_performance_metrics(
                Ytest_logits, Ptest_softmax, average="macro"
            )
            # [print(f'{key}: {value}\n') for key, value in summary_test['last'].items()]
            summary_test[str(cv_f + 1)]["last"]["per_case_prediction"] = Ptest

            # ###################### BEST MODEL
            model.load_weights(best_model_path)
            # get the per_slice classification
            Ptest_softmax = []
            for i in range(Xtest.shape[0]):
                Ptest_softmax.append(
                    model(np.expand_dims(Xtest[i, :, :, :], axis=0)).numpy()[0]
                )
            Ptest_softmax = np.array(Ptest_softmax)
            Ptest = np.argmax(Ptest_softmax, axis=-1)

            Ytest_logits = to_categorical(Ytest_classification, num_classes=2)
            summary_test[str(cv_f + 1)]["best"] = utilities.get_performance_metrics(
                Ytest_logits, Ptest_softmax, average="macro"
            )
            # [print(f'{key}: {value}\n') for key, value in summary_test['last'].items()]
            summary_test[str(cv_f + 1)]["best"]["per_case_prediction"] = Ptest

            ## SAVE TRAINING CURVES

            fig, ax = plt.subplots(figsize=(20, 15), nrows=2, ncols=1)
            # print training loss
            ax[0].plot(history.history["loss"], label="training loss")
            ax[0].plot(history.history["val_loss"], label="validation loss")
            ax[0].set_title(f"Test loss")
            ax[0].legend()
            # print training accuracy
            ax[1].plot(history.history["accuracy"], label="training accuracy")
            ax[1].plot(history.history["val_accuracy"], label="validation accuracy")
            ax[1].set_title(
                f'Test accuracy -> (last)  {summary_test[str(cv_f+1)]["last"]["overall_accuracy"]:0.3f}, (best) {summary_test[str(cv_f+1)]["best"]["overall_accuracy"]:0.3f}'
            )
            ax[1].legend()
            fig.savefig(os.path.join(save_model_path, "training_curves.png"))

            ## SAVE MODEL PORFORMANCE FOR for THIS fold
            for m in ["last", "best"]:
                filename = os.path.join(
                    args_dict["SAVE_PATH"],
                    setting,
                    f"fold_{str(cv_f+1)}",
                    f"{m}_summary_evaluation.txt",
                )
                accs = summary_test[str(cv_f + 1)][m]["overall_accuracy"] * 100
                np.savetxt(filename, [accs], fmt="%.4f")

            # SAVE PER METRICS AS CSV
            summary_file = os.path.join(
                args_dict["SAVE_PATH"],
                setting,
                f"fold_{str(cv_f+1)}",
                f"tabular_test_summary.csv",
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
                        model_version,
                        m,
                        cv_f + 1,
                        summary_test[str(cv_f + 1)][m]["overall_precision"],
                        summary_test[str(cv_f + 1)][m]["overall_recall"],
                        summary_test[str(cv_f + 1)][m]["overall_accuracy"],
                        summary_test[str(cv_f + 1)][m]["overall_f1-score"],
                        summary_test[str(cv_f + 1)][m]["overall_auc"],
                        summary_test[str(cv_f + 1)][m][
                            "matthews_correlation_coefficient"
                        ],
                    ]
                )
            writer.writerows(csv_rows)
            csv_file.close()
    ## SAVE SUMMARY FOR ALL THE FOLDS IN ONE FILE
    summary_file = os.path.join(
        args_dict["SAVE_PATH"], setting, f"tabular_test_summary.csv"
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
    for cv_f in range(args_dict["NBR_FOLDS"]):
        for m in ["last", "best"]:
            csv_rows.append(
                [
                    "tumor-vs-no_tumor",
                    2,
                    model_version,
                    m,
                    cv_f + 1,
                    summary_test[str(cv_f + 1)][m]["overall_precision"],
                    summary_test[str(cv_f + 1)][m]["overall_recall"],
                    summary_test[str(cv_f + 1)][m]["overall_accuracy"],
                    summary_test[str(cv_f + 1)][m]["overall_f1-score"],
                    summary_test[str(cv_f + 1)][m]["overall_auc"],
                    summary_test[str(cv_f + 1)][m]["matthews_correlation_coefficient"],
                ]
            )
    writer.writerows(csv_rows)
    csv_file.close()

# %%
