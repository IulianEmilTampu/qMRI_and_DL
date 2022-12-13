"""
Script that given a model and a series of images, plots and saved the GradCAM
images.

Steps
1 - get paths for the image data and the model
2 - load the model and get the information about the layers to use
3 - create data generators or take out images for plotting
4 - plot and save gradCAMS
"""

import os
import sys
import glob
import json
import warnings
import numpy as np
import argparse
import importlib
from pathlib import Path

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# local imports
import utilities
import models

## 1 - get paths
WORKING_FOLDER = "/home/iulta54/Code/P4_1-unsupervised"
IMG_DATASET_FOLDER = "/home/iulta54/Data/Gliom"
ANNOTATION_DATASET_FOLDER = "/home/iulta54/Data/Gliom/All_Annotations"
BRAIN_MASK_FOLDER = "/home/iulta54/Data/Gliom/All_Brain_masks"
MODEL_PATH = (
    "/home/iulta54/Code/P4_1-unsupervised/trained_models/Simple_CNN_5folds/qMRIGD"
)
N_GPU = "0"

# --------------------------------------
# set GPU
# --------------------------------------

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = N_GPU
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

devices = tf.config.list_physical_devices("GPU")

##1.1 - open the train_val_test_subject.json file and get path to the images
fold = 2
with open(
    os.path.join(os.path.dirname(MODEL_PATH), "train_val_test_subjects.json")
) as file:
    config = json.load(file)
    # take only the first fold
    config["train"] = config["train"][fold]
    config["validation"] = config["validation"][fold]

# build filename based on the model input specifications
test_gt_files = [os.path.join(ANNOTATION_DATASET_FOLDER, f) for f in config["test"]]
val_gt_files = [
    os.path.join(ANNOTATION_DATASET_FOLDER, f) for f in config["validation"]
]
train_gt_file = [os.path.join(ANNOTATION_DATASET_FOLDER, f) for f in config["train"]]

test_gt, validation_gt, train_gt = [], [], []
test_bm, validation_bm, train_bm = [], [], []

ds = "test"
for idx, gt in enumerate(test_gt_files):
    print(f"Woring on {ds} annotation {idx}/{len(test_gt_files)}\r", end="")
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

ds = "validation"
for idx, gt in enumerate(val_gt_files):
    print(f"Woring on {ds} annotation {idx}/{len(val_gt_files)}\r", end="")
    aus_annotations_archive = utilities.load_3D_data([gt])
    annotations = aus_annotations_archive["data_volumes"][:, :, :, :, 0]
    annotations = annotations.transpose(0, 3, 1, 2)
    annotations = annotations.reshape(
        annotations.shape[0] * annotations.shape[1],
        annotations.shape[2],
        annotations.shape[3],
    )
    annotations = annotations / np.max(annotations)
    validation_gt.append(to_categorical(annotations.astype("int"), 2))

validation_gt = np.concatenate(validation_gt, axis=0)

ds = "training"
for idx, gt in enumerate(train_gt_file):
    print(f"Woring on {ds} annotation {idx}/{len(train_gt_file)}\r", end="")
    aus_annotations_archive = utilities.load_3D_data([gt])
    annotations = aus_annotations_archive["data_volumes"][:, :, :, :, 0]
    annotations = annotations.transpose(0, 3, 1, 2)
    annotations = annotations.reshape(
        annotations.shape[0] * annotations.shape[1],
        annotations.shape[2],
        annotations.shape[3],
    )
    annotations = annotations / np.max(annotations)
    train_gt.append(to_categorical(annotations.astype("int"), 2))

train_gt = np.concatenate(train_gt, axis=0)

# get which data configuration the model has been trained on based on the name
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

input_configuration = os.path.basename(MODEL_PATH)
input_configuratio_idx = [
    k for k, value in combination_dict.items() if value[0] == input_configuration
][0]

modalities = combination_dict[input_configuratio_idx][1]

# check if all modaliites are available, if not print warning and move to the next setting
check_modalities = [
    m for m in modalities if os.path.isdir(os.path.join(IMG_DATASET_FOLDER, m)) == False
]
if check_modalities:
    print(
        f'\n {"#"*10} \n ATTENTION! \n Not all modalities are available {check_modalities}! Check dataset folder. \n Moving to the next setting. \n {"#"*10}'
    )
else:
    # create test, validation and training path for all the modalities
    file_dict = {"test": {}, "train": {}, "validation": {}}
    for ds in ["test", "train", "validation"]:
        for s in config[ds]:
            subject_number = s.split("_")[-1].split(".")[0]
            file_dict[ds][subject_number] = {}
            for m in modalities:
                file_dict[ds][subject_number][m] = os.path.join(
                    IMG_DATASET_FOLDER,
                    m,
                    "_".join(m.split("_")[1::]) + "_" + s.split("_")[-1],
                )

# finally load data
test_img, validation_img, train_img = [], [], []

# load test data
ds = "test"
for s in file_dict[ds]:
    aus_subject = []
    for m in file_dict[ds][s]:
        print(f"Working on subject dataset {ds}, subject {s}, modality {m} \r", end="")
        aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
    test_img.append(np.concatenate(aus_subject, axis=3))
test_img = np.concatenate(test_img, axis=0)

ds = "validation"
for s in file_dict[ds]:
    aus_subject = []
    for m in file_dict[ds][s]:
        print(f"Working on subject dataset {ds}, subject {s}, modality {m} \r", end="")
        aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
    validation_img.append(np.concatenate(aus_subject, axis=3))
validation_img = np.concatenate(validation_img, axis=0)

ds = "train"
for s in file_dict[ds]:
    aus_subject = []
    for m in file_dict[ds][s]:
        print(f"Working on subject dataset {ds}, subject {s}, modality {m} \r", end="")
        aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
    train_img.append(np.concatenate(aus_subject, axis=3))
train_img = np.concatenate(train_img, axis=0)

## 2 - load the model and get the information about the layers to use

model = tf.keras.models.load_model(
    os.path.join(MODEL_PATH, f"fold_{fold+1}", "last_model"), compile=False
)
# and load best model weight
model.load_weights(os.path.join(MODEL_PATH, f"fold_{fold+1}", "best_model_weights", ""))

## 2.2 - work on getting the layers to perform GradCAM on
importlib.reload(utilities)
print("\nPlotting GradCAM")

# all conv layers
name_layers = []
print("Looking for 2D conv layers...")
for layer in model.layers:
    if "conv" in layer.name:
        if "conv2d" in layer.name:
            # here no conv blocks
            name_layers.append(layer.name)

print("Found {} layers -> {}".format(len(name_layers), name_layers))

# get predictions on the a dataset data (and groud truth)
# change here based on the dataset one wants to plot
dataset_type = "test"
CAM_img_dataset = test_img
CAM_annotation_dataset = test_gt
labels = (np.sum(CAM_annotation_dataset[:, :, :, 1], axis=(1, 2)) > 0).astype(int)

pred_logits = []
heatmap_raw = []
heatmap_rgb = []

# compute activation maps for each image and each network layer
for i in range(CAM_img_dataset.shape[0]):
    print(
        f"Computing activation maps for each layer for the predicted class: {i+1}/{CAM_img_dataset.shape[0]} \r",
        end="",
    )
    image = np.expand_dims(CAM_img_dataset[i], axis=0)
    # get model prediction for this image
    img_pred_logits = model(image)
    # get model classification
    c = np.argmax(img_pred_logits)
    # save pred classification for mater
    pred_logits.append(img_pred_logits)
    # for all the images, compute heatmap for all the layers
    heatmap_raw.append([])
    heatmap_rgb.append([])
    for nl in name_layers:
        cam = utilities.gradCAM(model, c, layerName=nl)
        aus_raw, aus_rgb = cam.compute_heatmap(image)
        heatmap_raw[i].append(aus_raw)
        heatmap_rgb[i].append(aus_rgb)
print("\nDone.")

## plot images
layers_to_print = 2  # this specifies how many layers to print from the last
n_samples_per_image = 1
n_images = CAM_img_dataset.shape[0] // n_samples_per_image
fix_image = True


for i in range(n_images):
    print(f"Creating figure {i+1:3d}/{n_images}\r", end="")

    # greate figure
    # set different axis aspect ratios. The last axes is for the heat map -> smaller axes
    aus = [1 for i in range(len(range(layers_to_print)) + 1)]
    aus[-1] = 0.1
    gridspec = {"width_ratios": aus}
    fig, axes = plt.subplots(
        nrows=n_samples_per_image,
        ncols=len(range(layers_to_print)) + 1,
        figsize=(layers_to_print * 5, n_samples_per_image * 2),
    )
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, axis=0)
    # fig.suptitle('Consecutive activation maps', fontsize=16)

    # fill in all axes
    for j in range(n_samples_per_image):
        idx = i * n_samples_per_image + j
        if idx >= CAM_img_dataset.shape[0]:
            break

        # original image
        original_image = np.squeeze(CAM_img_dataset[idx, :, :, 0])
        if fix_image == True:
            original_image = np.rot90(original_image, k=1)
        axes[j, 0].imshow(original_image, cmap="gray", interpolation=None)
        pred_str = [f"{i:0.2f}" for i in pred_logits[idx][0]]
        if labels[idx] == np.argmax(pred_logits[idx]):
            axes[j, 0].set_title(f"gt {labels[idx]} - pred {pred_str}", color="g")
        else:
            axes[j, 0].set_title(f"gt {labels[idx]} - pred {pred_str}", color="r")

        axes[j, 0].set_xticks([])
        axes[j, 0].set_yticks([])

        # layer heatmaps
        for idx1, idx2 in enumerate(reversed(range(layers_to_print))):
            heat_map_image = heatmap_raw[idx][-(idx2 + 1)] / 255
            layer_name = name_layers[-(idx2 + 1)]
            if fix_image == True:
                heat_map_image = np.rot90(heat_map_image, k=1)
            im = axes[j, idx1 + 1].imshow(
                heat_map_image, cmap="jet", vmin=0, vmax=1, interpolation=None
            )
            axes[j, idx1 + 1].set_title(f"layer {layer_name}")
            axes[j, idx1 + 1].set_xticks([])
            axes[j, idx1 + 1].set_yticks([])

        # add colorbar
        # cax = axes[j, -1]
        # plt.colorbar(im, cax=cax)
        # plt.tight_layout()

        aspect = 20
        pad_fraction = 0.5

        divider = make_axes_locatable(axes[j, -1])
        width = axes_size.AxesY(axes[j, -1], aspect=1.0 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.colorbar(im, cax=cax)

    # fig.savefig(os.path.join(save_path, 'fold_'+str(fold)+'_activationMap_forConsecutiveLayers_%03d.pdf' % i), bbox_inches='tight', dpi = 100)
    save_path_cams = os.path.join(
        MODEL_PATH, f"fold_{fold+1}", f"GradCAM_{dataset_type}"
    )
    Path(save_path_cams).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        os.path.join(save_path_cams, "activationMap_forConsecutiveLayers_%03d.png" % i),
        bbox_inches="tight",
        dpi=100,
    )
    plt.close(fig)
