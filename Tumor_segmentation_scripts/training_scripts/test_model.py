# %% ########################################################################################
"""
Script that given a model and the configuration file used for training, tests
the model on the test dataset and save the information in a txt file

Steps
1 - get paths for the image data and the model
2 - for every fold, load the last and best models
3 - test models and save fold information
"""

from multiprocessing.sharedctypes import Value
import os
import sys
import glob
import json
import csv
import warnings
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# %matplotlib inline

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning)

# local imports
import utilities

# %%
su_debug_flag = False

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
        "--DATASET_FOLDER",
        required=True,
        help="Provide the Image Dataset Folder where the folders for each modality are located.",
    )
    parser.add_argument(
        "-af",
        "--ANNOTATION_FOLDER",
        required=True,
        help="Provide the Annotation  Folder where annotations are located.",
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
        "-aic",
        "--ANATOMICAL_IMG_CHANNEL",
        required=False,
        default=0,
        help="Provide which channels of the input configuration to use for the plotting.",
    )
    parser.add_argument(
        "-g",
        "--GPU",
        required=False,
        default=0,
        help="Provide the GPU number to use for training.",
    )

    args_dict = dict(vars(parser.parse_args()))
    args_dict["NFOLDS"] = len(
        glob.glob(os.path.join(args_dict["MODEL_PATH"], "fold_*", ""))
    )
    args_dict["ANATOMICAL_IMG_CHANNEL"] = int(args_dict["ANATOMICAL_IMG_CHANNEL"])

else:
    # # # # # # # # # # # # # # DEBUG
    print("Running in debug mode.")
    args_dict = {}
    args_dict["WORKING_FOLDER"] = "/flush/iulta54/Research/P4-qMRI"
    args_dict[
        "DATASET_FOLDER"
    ] = "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality"
    args_dict[
        "ANNOTATION_FOLDER"
    ] = "/flush/iulta54/Research/Data/qMRI_data/qMRI_per_modality/All_Annotations"
    args_dict[
        "TRAIN_VALIDATION_TEST_FILE"
    ] = "/flush/iulta54/Research/P4-qMRI/trained_models/Simple_2DUNet_rkv_5_fold5_lr0.001_batch3_cv_repetition_5_seed_1238/train_val_test_subjects.json"
    args_dict[
        "MODEL_PATH"
    ] = "/flush/iulta54/Research/P4-qMRI/trained_models/Simple_2DUNet_rkv_5_fold5_lr0.001_batch3_cv_repetition_5_seed_1238/qMRIGD"
    args_dict["GPU"] = "0"
    args_dict["NFOLDS"] = len(
        glob.glob(os.path.join(args_dict["MODEL_PATH"], "fold_*", ""))
    )
    args_dict["ANATOMICAL_IMG_CHANNEL"] = 0

# build save path
args_dict["SAVE_PATH"] = os.path.join(args_dict["MODEL_PATH"], "Summary_test")
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

combination_dict = {
    "0": ["BRATS", ["All_T1FLAIR_GD", "All_T1FLAIR", "All_T2FLAIR", "All_T2"]],
    "1": ["qMRI", ["All_qMRI_T1", "All_qMRI_T2", "All_qMRI_PD"]],
    "2": ["qMRI_GD", ["All_qMRI_T1_GD", "All_qMRI_T2_GD", "All_qMRI_PD_GD"]],
    "3": [
        "BRATS-qMRI",
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
        "BRATS-qMRI_GD",
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
        "qMRI-qMRI_GD",
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
        " BRATS-qMRI-qMRI_GD",
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
        "BRATS-Delta_Tumor_border",
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
os.environ["CUDA_VISIBLE_DEVICES"] = args_dict["GPU"]

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

devices = tf.config.list_physical_devices("GPU")

if devices:
    print(f'Running training on GPU # {args_dict["GPU"]} \n')
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
    os.path.join(args_dict["ANNOTATION_FOLDER"], f) for f in config["test"]
]
# load annotation data
test_gt = []
test_gt_subject_slide_id = []

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
    test_gt_subject_slide_id.extend(
        [
            "_".join([os.path.basename(gt).split(".")[0], f"slice_{i+1}"])
            for i in range(annotations.shape[0])
        ]
    )

test_gt = np.concatenate(test_gt, axis=0)

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
    if os.path.isdir(os.path.join(args_dict["DATASET_FOLDER"], m)) == False
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
                args_dict["DATASET_FOLDER"],
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
            f"Working on subject dataset {ds}, subject {s}, modality {m}          \r",
            end="",
        )
        aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
    test_img.append(np.concatenate(aus_subject, axis=3))
test_img = np.concatenate(test_img, axis=0)

print("\n")

# %% - LOOP THOURH ALL THE FOLDS AND TEST LAST AND BEST MODEL
"""
Here we save for each fold we save the anatomical image, the segmentation, the Dice and accuracy for each test sample
"""


def dice(im1, im2, empty_score=1.0, per_class=False):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    per_class : bool
        Returns the dice score for each class (channels in the img). The class channel should
        be the last one.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if per_class:
        # return dice score for each class (channel in the im)
        dice_scores = []
        for c in range(im1.shape[-1]):
            im_sum = im1[:, :, c].sum() + im2[:, :, c].sum()
            if im_sum == 0:
                dice_scores.append(empty_score)
            else:
                intersection = np.logical_and(im1[:, :, c], im2[:, :, c])
                dice_scores.append(2.0 * intersection.sum() / im_sum)
        return dice_scores
    else:
        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return empty_score

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        return 2.0 * intersection.sum() / im_sum


last_best_model_names = {"last": "last_model.tf", "best": "best_model.tf"}

summary_test = {}

for f in range(args_dict["NFOLDS"]):
    # load last model (for the best model, just load weights)
    model_path = os.path.join(
        args_dict["MODEL_PATH"], f"fold_{f+1}", last_best_model_names["last"]
    )
    # initialize where to save the test performance
    summary_test[str(f + 1)] = {"best": [], "last": []}
    for model_version in ["last", "best"]:
        try:
            model_path = os.path.join(
                args_dict["MODEL_PATH"],
                f"fold_{f+1}",
                last_best_model_names[model_version],
                "",
            )

            model = tf.keras.models.load_model(model_path, compile=False)
        except:
            print(f"Could not load best model for fold {f+1}. Given {model_path}")

        # if model is present, test it
        for i in range(test_img.shape[0]):
            print(
                f"Working on {model_version} model fold {f+1} ({i+1:3d}/{test_img.shape[0]})\r",
                end="",
            )
            model_segmentation = model(
                np.expand_dims(test_img[i, :, :, :], axis=0)
            ).numpy()[0]
            model_segmentation[model_segmentation >= 0.5] = 1
            model_segmentation[model_segmentation < 0.5] = 0

            dice_coef = dice(test_gt[i, :, :, 1], model_segmentation[:, :, 1])
            dict_test_sample = {
                "subject_ID": test_gt_subject_slide_id[i],
                "anatomical_image": test_img[i, :, :, :],
                "ground_truth": test_gt[i, :, :, :],
                "model_segmentation": model_segmentation,
                "Dice": dice(test_gt[i, :, :, 1], model_segmentation[:, :, 1]),
                "per_class_Dice": dice(
                    test_gt[i, :, :, :], model_segmentation[:, :, :], per_class=True
                ),
                "accuracy": 0,
            }
            # save info for this test sample for this fold
            summary_test[str(f + 1)][model_version].append(dict_test_sample)
    # if f == 0:
    #     break
print()

# %% 3 - PRINT SUMMARY TEST
metric_to_print = "Dice"
print(f"METRIC : {metric_to_print}")
for f in range(len(summary_test)):
    mean_metric_last = np.mean(
        [
            summary_test[str(f + 1)]["last"][i][metric_to_print]
            for i in range(len(summary_test[str(f + 1)]["last"]))
        ]
    )
    mean_metric_best = np.mean(
        [
            summary_test[str(f + 1)]["best"][i][metric_to_print]
            for i in range(len(summary_test[str(f + 1)]["best"]))
        ]
    )
    print(
        f"Fold {f+1} -> last model:{mean_metric_last:0.3f}, best model:{mean_metric_best:0.3f}"
    )

# %% DEFUNE PLOTTING UTILITY and plot images for every fold
def plot_segmentation_comparison(
    anatomical_img: np.array,
    ground_truth: np.array,
    model_segmentation: np.array,
    dice: np.array,
    per_class_dice: np.array,
    save_figure=False,
    save_name=None,
):
    def fix_img_for_plot(img):
        return np.fliplr(np.rot90(img, k=3))

    # plot settings
    title_font_size = 10

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 24), dpi=300)
    # plot anatomical image
    ax = axes[0]
    ax.imshow(fix_img_for_plot(anatomical_img), cmap="gray")
    ax.set_title(
        f"Anatomical image",
        fontsize=title_font_size,
    )
    ax.set_axis_off()

    # plot anatomical image with GT overlayed
    ax = axes[1]
    gt_mask = np.copy(ground_truth)
    gt_mask[gt_mask == 0] = np.nan
    ax.imshow(fix_img_for_plot(anatomical_img), cmap="gray", interpolation=None)
    ax.imshow(
        fix_img_for_plot(gt_mask), cmap="hsv", vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_title("GT overlayed", fontsize=title_font_size)
    ax.set_axis_off()

    # plot plot anatomical image with segmentation overlayed
    ax = axes[2]
    seg_mask = np.copy(model_segmentation)
    seg_mask[seg_mask == 0] = np.nan
    ax.imshow(fix_img_for_plot(anatomical_img), cmap="gray", interpolation=None)
    ax.imshow(
        fix_img_for_plot(seg_mask), cmap="hsv", vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_title("Segmentation overlayed", fontsize=title_font_size)
    ax.set_axis_off()

    # plot overlap segmentation
    ax = axes[3]
    ax.imshow(fix_img_for_plot(anatomical_img), cmap="gray")
    # compute tp, fp, tp, fn
    tp = (model_segmentation + ground_truth) / 2
    tp[tp < 1] = np.nan
    fp = model_segmentation - ground_truth
    fp[fp < 1] = np.nan
    fn = ground_truth - model_segmentation
    fn[fn < 1] = np.nan
    cmap = "hsv"
    ax.imshow(fix_img_for_plot(tp), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.imshow(
        fix_img_for_plot(fp * 2), cmap=cmap, vmin=0, vmax=3, interpolation="nearest"
    )
    ax.imshow(
        fix_img_for_plot(fn * 3), cmap=cmap, vmin=0, vmax=3, interpolation="nearest"
    )
    ax.set_title(
        f"Dice : {dice:0.3f}\nper class Dice\n{per_class_dice}",
        fontsize=title_font_size,
    )
    ax.set_axis_off()

    # save figure
    if save_figure:
        fig.savefig(
            save_name + ".png",
            bbox_inches="tight",
            dpi=300,
            facecolor=fig.get_facecolor(),
            transparent=True,
        )
        fig.savefig(
            save_name + ".pdf",
            bbox_inches="tight",
            dpi=300,
            facecolor=fig.get_facecolor(),
            transparent=True,
        )
        plt.close()
    else:
        plt.show()


# %% SAVE IMAGES FOR EVERY FOLD SEPARATELY
print("Saving individial samples for each fold images...")
for f in range(args_dict["NFOLDS"]):
    for model_version in ["best", "last"]:
        for i in range(len(test_gt_subject_slide_id)):
            # get img, gt and seg
            img = np.copy(
                summary_test[str(f + 1)][model_version][i]["anatomical_image"][
                    :, :, args_dict["ANATOMICAL_IMG_CHANNEL"]
                ]
            )
            gt = np.copy(
                summary_test[str(f + 1)][model_version][i]["ground_truth"][:, :, 1]
            )
            seg = np.copy(
                summary_test[str(f + 1)][model_version][i]["model_segmentation"][
                    :, :, 1
                ]
            )
            # build image name
            Path(os.path.join(args_dict["SAVE_PATH"], f"fold_{f+1}")).mkdir(
                parents=True, exist_ok=True
            )
            fig_name = os.path.join(
                args_dict["SAVE_PATH"],
                f"fold_{f+1}",
                f'{model_version}_model_ch_{args_dict["ANATOMICAL_IMG_CHANNEL"]}_{summary_test[str(f+1)][model_version][i]["subject_ID"]}',
            )

            per_class_dices = [
                np.round(
                    summary_test[str(f + 1)][model_version][i]["per_class_Dice"][j], 3
                )
                for j in range(
                    len(summary_test[str(f + 1)][model_version][i]["per_class_Dice"])
                )
            ]
            # make figure
            plot_segmentation_comparison(
                img,
                gt,
                seg,
                dice=summary_test[str(f + 1)][model_version][i]["Dice"],
                per_class_dice=per_class_dices,
                save_figure=True,
                save_name=fig_name,
            )
print("Done!")
# %% WORK ON THE ENSAMBLED PREDICTION (MAJORITY VOUTING)
print("Saving ensemble images...")
save_figure = True
summary_test["ensemble_seg"] = {"best": [], "last": []}
for model_version in ["best", "last"]:
    for i in range(len(test_gt_subject_slide_id)):
        # for each test image, collect the predictions from each fold
        aggregated_seg = [
            np.copy(summary_test[str(f + 1)][model_version][i]["model_segmentation"])
            for f in range(args_dict["NFOLDS"])
        ]
        aggregated_seg = np.mean(np.array(aggregated_seg), axis=0)
        aggregated_seg[aggregated_seg >= 0.5] = 1
        aggregated_seg[aggregated_seg < 0.5] = 0
        # compute metrics
        aggregated_dice = dice(
            summary_test[str(1)][model_version][i]["ground_truth"][:, :, 1],
            aggregated_seg[:, :, 1],
        )
        aggregared_dice_per_class = dice(
            summary_test[str(f + 1)][model_version][i]["ground_truth"],
            aggregated_seg,
            per_class=True,
        )
        # save information
        dict_test_sample = {
            "subject_ID": test_gt_subject_slide_id[i],
            "anatomical_image": np.copy(
                summary_test[str(1)][model_version][i]["anatomical_image"]
            ),
            "ground_truth": np.copy(
                summary_test[str(1)][model_version][i]["ground_truth"]
            ),
            "model_segmentation": aggregated_seg,
            "Dice": aggregated_dice,
            "per_class_Dice": aggregared_dice_per_class,
            "accuracy": 0,
        }
        summary_test["ensemble_seg"][model_version].append(dict_test_sample)


for model_version in ["best", "last"]:
    for i in range(len(test_gt_subject_slide_id)):
        # get img, gt and seg
        img = np.copy(
            summary_test["ensemble_seg"][model_version][i]["anatomical_image"][
                :, :, args_dict["ANATOMICAL_IMG_CHANNEL"]
            ]
        )
        gt = np.copy(
            summary_test["ensemble_seg"][model_version][i]["ground_truth"][:, :, 1]
        )
        seg = np.copy(
            summary_test["ensemble_seg"][model_version][i]["model_segmentation"][
                :, :, 1
            ]
        )
        # build image name
        Path(os.path.join(args_dict["SAVE_PATH"], "Ensemble_pred")).mkdir(
            parents=True, exist_ok=True
        )
        fig_name = os.path.join(
            args_dict["SAVE_PATH"],
            "Ensemble_pred",
            f'{model_version}_model_ch_{args_dict["ANATOMICAL_IMG_CHANNEL"]}_{summary_test[str(f+1)][model_version][i]["subject_ID"]}',
        )

        per_class_dices = [
            np.round(
                summary_test["ensemble_seg"][model_version][i]["per_class_Dice"][j], 3
            )
            for j in range(
                len(summary_test["ensemble_seg"][model_version][i]["per_class_Dice"])
            )
        ]
        # make figure
        plot_segmentation_comparison(
            img,
            gt,
            seg,
            dice=summary_test["ensemble_seg"][model_version][i]["Dice"],
            per_class_dice=per_class_dices,
            save_figure=True,
            save_name=fig_name,
        )
print("Done!")
# %% SAVE INFORMATION FOR EACH FOLD AND SUBJECT SLICE SEPARATELY
print("Saving per_test_case metrics for each fold...")

for f in range(args_dict["NFOLDS"]):
    summary_file = os.path.join(
        args_dict["MODEL_PATH"],
        f"fold_{str(f+1)}",
        f"fold_tabular_test_summary_per_slice_case.csv",
    )
    csv_file = open(summary_file, "w")
    writer = csv.writer(csv_file)
    csv_header = [
        "task",
        "nbr_classes",
        "model_type",
        "input_configuration",
        "model_version",
        "fold",
        "test_ID",
        "Dice",
        "background_Dice",
        "tumor_class_Dice",
    ]
    writer.writerow(csv_header)
    # build rows to save in the csv file
    csv_rows = []
    for model_version in ["best", "last"]:
        for i in range(len(test_gt_subject_slide_id)):
            csv_rows.append(
                [
                    "tumor_segmentation",
                    2,
                    "2D_UNet",
                    os.path.basename(args_dict["MODEL_PATH"]),
                    model_version,
                    f + 1,
                    summary_test[str(f + 1)][model_version][i]["subject_ID"],
                    summary_test[str(f + 1)][model_version][i]["Dice"],
                    summary_test[str(f + 1)][model_version][i]["per_class_Dice"][0],
                    summary_test[str(f + 1)][model_version][i]["per_class_Dice"][1],
                ]
            )
    writer.writerows(csv_rows)
    csv_file.close()

# %% SAVE INFORMATION FOR EACH FOLD (MEAN OVER SLICES)
print("Saving summary metrics for each fold...")
for f in range(args_dict["NFOLDS"]):
    summary_file = os.path.join(
        args_dict["MODEL_PATH"], f"fold_{str(f+1)}", f"fold_tabular_test_summary.csv"
    )
    csv_file = open(summary_file, "w")
    writer = csv.writer(csv_file)
    csv_header = [
        "task",
        "nbr_classes",
        "model_type",
        "input_configuration",
        "model_version",
        "fold",
        "Dice",
        "std_Dice",
        "mean_background_Dice",
        "std_background_Dice",
        "mean_tumor_class_Dice",
        "std_tumor_class_Dice",
    ]
    writer.writerow(csv_header)
    # build rows to save in the csv file
    csv_rows = []
    for model_version in ["best", "last"]:
        mean_Dice = np.mean(
            [
                summary_test[str(f + 1)][model_version][i]["Dice"]
                for i in range(len(summary_test[str(f + 1)][model_version]))
            ]
        )
        std_Dice = np.std(
            [
                summary_test[str(f + 1)][model_version][i]["Dice"]
                for i in range(len(summary_test[str(f + 1)][model_version]))
            ]
        )

        per_class_mean_Dice = np.mean(
            [
                summary_test[str(f + 1)][model_version][i]["per_class_Dice"]
                for i in range(len(summary_test[str(f + 1)][model_version]))
            ],
            axis=0,
        )
        per_class_std_Dice = np.std(
            [
                summary_test[str(f + 1)][model_version][i]["per_class_Dice"]
                for i in range(len(summary_test[str(f + 1)][model_version]))
            ],
            axis=0,
        )

        csv_rows.append(
            [
                "tumor_segmentation",
                2,
                "2D_UNet",
                os.path.basename(args_dict["MODEL_PATH"]),
                model_version,
                f + 1,
                mean_Dice,
                std_Dice,
                per_class_mean_Dice[0],
                per_class_std_Dice[0],
                per_class_mean_Dice[1],
                per_class_std_Dice[1],
            ]
        )
    writer.writerows(csv_rows)
    csv_file.close()

# %% SAVE INFORMATION FOR EACH FOLD (MEAN OVER SLICES)
import copy

print("Saving overall metrics for all the folds...")

summary_file = os.path.join(args_dict["MODEL_PATH"], f"tabular_test_summary.csv")
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
csv_header = [
    "task",
    "nbr_classes",
    "model_type",
    "input_configuration",
    "model_version",
    "fold",
    "Dice",
    "std_Dice",
    "mean_background_Dice",
    "std_background_Dice",
    "mean_tumor_class_Dice",
    "std_tumor_class_Dice",
]
writer.writerow(csv_header)

# build rows to save in the csv file
csv_rows = []
for f in range(args_dict["NFOLDS"]):
    for model_version in ["best", "last"]:
        mean_Dice = np.mean(
            [
                np.copy(summary_test[str(f + 1)][model_version][i]["Dice"])
                for i in range(len(summary_test[str(f + 1)][model_version]))
            ]
        )
        std_Dice = np.std(
            [
                np.copy(summary_test[str(f + 1)][model_version][i]["Dice"])
                for i in range(len(summary_test[str(f + 1)][model_version]))
            ]
        )

        per_class_mean_Dice = np.mean(
            [
                np.copy(summary_test[str(f + 1)][model_version][i]["per_class_Dice"])
                for i in range(len(summary_test[str(f + 1)][model_version]))
            ],
            axis=0,
        )
        per_class_std_Dice = np.std(
            [
                np.copy(summary_test[str(f + 1)][model_version][i]["per_class_Dice"])
                for i in range(len(summary_test[str(f + 1)][model_version]))
            ],
            axis=0,
        )

        csv_rows.append(
            [
                "tumor_segmentation",
                2,
                "2D_UNet",
                os.path.basename(args_dict["MODEL_PATH"]),
                model_version,
                f + 1,
                mean_Dice,
                std_Dice,
                per_class_mean_Dice[0],
                per_class_std_Dice[0],
                per_class_mean_Dice[1],
                per_class_std_Dice[1],
            ]
        )
# save also ensemble metric
for model_version in ["best", "last"]:
    # aus_dict = b = copy.deepcopy(summary_test)
    mean_Dice = np.mean(
        [
            np.copy(summary_test["ensemble_seg"][model_version][i]["Dice"])
            for i in range(len(test_gt_subject_slide_id))
        ]
    )
    std_Dice = np.std(
        [
            np.copy(summary_test["ensemble_seg"][model_version][i]["Dice"])
            for i in range(len(test_gt_subject_slide_id))
        ]
    )
    per_class_mean_Dice = np.mean(
        [
            np.copy(summary_test["ensemble_seg"][model_version][i]["per_class_Dice"])
            for i in range(len(test_gt_subject_slide_id))
        ],
        axis=0,
    )
    per_class_std_Dice = np.std(
        [
            np.copy(summary_test["ensemble_seg"][model_version][i]["per_class_Dice"])
            for i in range(len(test_gt_subject_slide_id))
        ],
        axis=0,
    )

    csv_rows.append(
        [
            "tumor_segmentation",
            2,
            "2D_UNet",
            os.path.basename(args_dict["MODEL_PATH"]),
            model_version,
            "ensemble",
            mean_Dice,
            std_Dice,
            per_class_mean_Dice[0],
            per_class_std_Dice[0],
            per_class_mean_Dice[1],
            per_class_std_Dice[1],
        ]
    )

writer.writerows(csv_rows)
csv_file.close()

print("Done!")
print("Finished testing")
