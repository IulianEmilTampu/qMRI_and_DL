'''
Script that given a model and a series of images, plots and saved the occlusion maps
images.

Steps
1 - get paths for the image data and the model
2 - load the model
4 - generates the occusion map for a selected dataset
5 - plots the occusion maps
'''

import os
import sys
import glob
import json
import cv2
import warnings
import numpy as np
import argparse
import importlib
from pathlib import Path
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['AUTOGRAPH_VERBOSITY'] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# local imports
import utilities
import models

## utilities

# colorbar utility
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

## 1 - get paths
WORKING_FOLDER = "/home/iulta54/Code/P4_1-unsupervised"
IMG_DATASET_FOLDER = "/home/iulta54/Data/Gliom"
ANNOTATION_DATASET_FOLDER = "/home/iulta54/Data/Gliom/All_Annotations"
BRAIN_MASK_FOLDER = "/home/iulta54/Data/Gliom/All_Brain_masks"
MODEL_PATH = "/home/iulta54/Code/P4_1-unsupervised/trained_models/Simple_CNN_5folds/BRATS"
N_GPU = "1"

#--------------------------------------
# set GPU
#--------------------------------------

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]=N_GPU;
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
devices = tf.config.list_physical_devices('GPU')

##1.1 - open the train_val_test_subject.json file and get path to the images
fold = 2
with open(os.path.join(os.path.dirname(MODEL_PATH), 'train_val_test_subjects.json')) as file:
    config = json.load(file)
    # take only the first fold
    config['train'] = config['train'][fold]
    config['validation'] = config['validation'][fold]

# build filename based on the model input specifications
test_gt_files =  [os.path.join(ANNOTATION_DATASET_FOLDER, f) for f in config['test']]
val_gt_files = [os.path.join(ANNOTATION_DATASET_FOLDER, f) for f in config['validation']]
train_gt_file = [os.path.join(ANNOTATION_DATASET_FOLDER, f) for f in config['train']]

test_gt, validation_gt, train_gt = [], [], []
test_bm, validation_bm, train_bm = [], [], []

ds = 'test'
for idx, gt in enumerate(test_gt_files):
    print(f'Woring on {ds} annotation {idx+1}/{len(test_gt_files)}\r', end='')
    aus_annotations_archive = utilities.load_3D_data([gt])
    annotations = aus_annotations_archive['data_volumes'][:,:,:,:,0]
    annotations = annotations.transpose(0,3,1,2)
    annotations = annotations.reshape(annotations.shape[0]*annotations.shape[1], annotations.shape[2], annotations.shape[3])
    annotations = annotations / np.max(annotations)
    test_gt.append(to_categorical(annotations.astype('int'),2))

test_gt = np.concatenate(test_gt, axis=0)

ds = 'validation'
for idx, gt in enumerate(val_gt_files):
    print(f'Woring on {ds} annotation {idx+1}/{len(val_gt_files)}\r', end='')
    aus_annotations_archive = utilities.load_3D_data([gt])
    annotations = aus_annotations_archive['data_volumes'][:,:,:,:,0]
    annotations = annotations.transpose(0,3,1,2)
    annotations = annotations.reshape(annotations.shape[0]*annotations.shape[1], annotations.shape[2], annotations.shape[3])
    annotations = annotations / np.max(annotations)
    validation_gt.append(to_categorical(annotations.astype('int'),2))

validation_gt = np.concatenate(validation_gt, axis=0)

ds = 'training'
for idx, gt in enumerate(train_gt_file):
    print(f'Woring on {ds} annotation {idx+1}/{len(train_gt_file)}\r', end='')
    aus_annotations_archive = utilities.load_3D_data([gt])
    annotations = aus_annotations_archive['data_volumes'][:,:,:,:,0]
    annotations = annotations.transpose(0,3,1,2)
    annotations = annotations.reshape(annotations.shape[0]*annotations.shape[1], annotations.shape[2], annotations.shape[3])
    annotations = annotations / np.max(annotations)
    train_gt.append(to_categorical(annotations.astype('int'),2))

train_gt = np.concatenate(train_gt, axis=0)


combination_dict = {'0':['BRATS',
                            ['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2'],
                            ['T1w_GD', 'T1w', 'T2w','FLAIR']],
                    '1':['qMRI',
                            ['All_qMRI_T1', 'All_qMRI_T2', 'All_qMRI_PD'],
                            ['qT1', 'qT2','qPD']],
                    '2':['qMRIGD',
                            ['All_qMRI_T1_GD', 'All_qMRI_T2_GD', 'All_qMRI_PD_GD'],
                            ['qT1_GD', 'qT2_GD','qPD_GD']],
                    '3':['BRATS-qMRI',
                            ['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2', 'All_qMRI_T1', 'All_qMRI_T2', 'All_qMRI_PD'],
                            ['T1w_GD', 'T1w', 'T2w','FLAIR', 'qT1', 'qT2','qPD']],
                    '4':['BRATS-qMRIGD',
                            ['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2', 'All_qMRI_T1_GD', 'All_qMRI_T2_GD', 'All_qMRI_PD_GD'],
                            ['T1w_GD', 'T1w', 'T2w','FLAIR','qT1_GD', 'qT2_GD','qPD_GD']],
                    '5':['qMRI-qMRIGD',
                            ['All_qMRI_T1', 'All_qMRI_T2', 'All_qMRI_PD', 'All_qMRI_T1_GD', 'All_qMRI_T2_GD', 'All_qMRI_PD_GD'],
                            ['qT1', 'qT2','qPD', 'qT1_GD', 'qT2_GD','qPD_GD']],
                    '6':['BRATS-qMRI-qMRIGD',
                            ['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2', 'All_qMRI_T1', 'All_qMRI_T2', 'All_qMRI_PD', 'All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2', 'All_qMRI_T1_GD', 'All_qMRI_T2_GD', 'All_qMRI_PD_GD'],
                              ['T1w_GD', 'T1w', 'T2w','FLAIR', 'qT1', 'qT2','qPD', 'qT1_GD', 'qT2_GD','qPD_GD']],
                    '7':['T1WGD',
                            ['All_T1FLAIR_GD'],
                            ['T1w_GD']],
                    '8':['Delta_Tumor_border',
                            ['All_delta_border_R1_GD', 'All_delta_border_R2_GD', 'All_delta_border_PD_GD']],
                    '9':['BRATS-Delta_Tumor_border',
                            ['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2','All_delta_border_R1_GD', 'All_delta_border_R2_GD', 'All_delta_border_PD_GD']],
                    '10':['T1wGD_T1GD_T2GD',
                            ['All_T1FLAIR_GD', 'All_qMRI_T1_GD', 'All_qMRI_T2_GD'],
                            ['T1w_GD', 'qT1_GD', 'qT2_GD']],
                    '11':['qT1GD',
                            ['All_qMRI_T1_GD'],
                            ['qT1_GD']],
                    '12':['qT2GD',
                            ['All_qMRI_T2_GD'],
                            ['qT2_GD']],
                    '13':['qT1GD-qT2GD',
                            ['All_qMRI_T1_GD','All_qMRI_T2_GD'],
                            ['qT1_GD', 'qT2_GD']],
                    '14':['T1WGD-qT1GD',
                            ['All_T1FLAIR_GD','All_qMRI_T1_GD'],
                            ['T1w_GD', 'qT1_GD']],
                    '15':['qPDDG',
                            ['All_qMRI_PD_GD']],
                    '16':['qPD',['All_qMRI_PD']],
                    '17':['T1w-T1wGD-T2w',['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2']],
                    '18':['T2w',['All_T2']],
                    '19':['FLAIR',['All_T2FLAIR']],
                    }

input_configuration = os.path.basename(MODEL_PATH)
input_configuratio_idx = [k for k, value in combination_dict.items() if value[0] == input_configuration][0]

modalities = combination_dict[input_configuratio_idx][1]
modalities_names = combination_dict[input_configuratio_idx][2]

# check if all modaliites are available, if not print warning and move to the next setting
check_modalities = [m for m in modalities if os.path.isdir(os.path.join(IMG_DATASET_FOLDER, m))==False]
if check_modalities:
    print(f'\n {"#"*10} \n ATTENTION! \n Not all modalities are available {check_modalities}! Check dataset folder. \n Moving to the next setting. \n {"#"*10}')
else:
    # create test, validation and training path for all the modalities
    file_dict = {'test':{}, 'train':{}, 'validation':{}}
    for ds in ['test', 'train', 'validation']:
        for s in config[ds]:
            subject_number = s.split('_')[-1].split('.')[0]
            file_dict[ds][subject_number] = {}
            for m in modalities:
                file_dict[ds][subject_number][m] = os.path.join(IMG_DATASET_FOLDER, m, '_'.join(m.split('_')[1::]) + '_' + s.split('_')[-1])

# finally load data
test_img, validation_img, train_img = [], [], []

# load test data
ds = 'test'
for s in file_dict[ds]:
    aus_subject = []
    for m in file_dict[ds][s]:
        print(f'Working on subject dataset {ds}, subject {s}, modality {m} \r', end='')
        aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
    test_img.append(np.concatenate(aus_subject, axis=3))
test_img = np.concatenate(test_img, axis=0)

ds = 'validation'
for s in file_dict[ds]:
    aus_subject = []
    for m in file_dict[ds][s]:
        print(f'Working on subject dataset {ds}, subject {s}, modality {m} \r', end='')
        aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
    validation_img.append(np.concatenate(aus_subject, axis=3))
validation_img = np.concatenate(validation_img, axis=0)

ds = 'train'
for s in file_dict[ds]:
    aus_subject = []
    for m in file_dict[ds][s]:
        print(f'Working on subject dataset {ds}, subject {s}, modality {m} \r', end='')
        aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
    train_img.append(np.concatenate(aus_subject, axis=3))
train_img = np.concatenate(train_img, axis=0)

print('\n')

## 2 - load the model and get the information about the layers to use

model = tf.keras.models.load_model(os.path.join(MODEL_PATH, f'fold_{fold+1}','last_model'), compile=False)
# and load best model weight
model.load_weights(os.path.join(MODEL_PATH, f'fold_{fold+1}','best_model_weights',''))

## 3 create fundtion that give an image generates the occlusion map for it
def get_occlusion_maps(model, image, patch_size=10, patch_value=0, debug=False):
    '''
    Function that given a model and an image that can be infered by the model,
    creates the occlusion maps for all the classification outputs (one image per
    class).
    INPUTS
    model : tf.keras.model
        Trained model
    imges : numpy array with shape [1, width, hight, nbr_channels]
        Image that the model is able to infere on.
    patch_size : int
        Specifies how large is the size of the squared patch used to compute
        the occlusion map.
    patch_value : float
        Value to set the patch to

    OUTPUT
    occusion_maps :  numpy arrays of shape [n_classes, width, hight]
        The rray stores the difference in logits classification between
        the patched image and the not-patched image in the pixels that are patched.

    STEPS
    - compute the indexes to get the pathes from the image based on the
    patch size.
    - for all the patches
        # set the patch to patch_value
        # compute model prediction on the patched image
        # compute difference between logits in the patched image and the original
          image
        # store the value of the difference in the occusion_map
    '''

    '''
    To be implemented: checks for the validity of the variables
    '''

    # initiate variables
    n_classes = model.output.shape[-1]
    x_shape, y_shape = image.shape[1], image.shape[2]
    occusion_maps_logits = np.zeros((n_classes, x_shape, y_shape))
    occusion_maps_classification = np.zeros((x_shape, y_shape))

     # make the last index to be the width of the image -> the last patch smaller that the actual patch size
    row_indexes = list(range(0, x_shape, patch_size))
    row_indexes.append(x_shape)
    # same for the colunms
    col_indexes = list(range(0, y_shape, patch_size))
    col_indexes.append(y_shape)

    # compute whole image prediction logits
    whole_image_logits = model(image).numpy()[0]

    # loop through all the patches
    status_idx = 0
    total_patches = (len(row_indexes)-1)*(len(col_indexes)-1)
    for r_idx in range(1, len(row_indexes)):
        for c_idx in range(1, len(col_indexes)):
            status_idx += 1
            # print(f'Woring on patch {status_idx}/{len(row_indexes)*len(col_indexes)} {(patched_image.sum())} \r', end='')
            # patch image
            patched_image = np.array(image)
            patched_image[:, row_indexes[r_idx-1]:row_indexes[r_idx],col_indexes[c_idx-1]:col_indexes[c_idx],:] = patch_value

            if debug:
                print(f'Woring on {total_patches} patches ({100*r_idx/total_patches:0.2f}%) \r', end='')
                # print(f'Woring on patch {status_idx}/{(len(row_indexes)-1)*(len(col_indexes)-1)} {(image.sum())} {(patched_image.sum())} \r', end='')
            # compute prediction on the patched image
            patch_logits = model(patched_image).numpy()[0]
            patch_classification = np.argmax(patch_logits, axis=-1)
            # compute logits difference
            logits_diff = np.abs(whole_image_logits - patch_logits)
            # put the values in the occluded patch
            for c in range(n_classes):
                occusion_maps_logits[c, row_indexes[r_idx-1]:row_indexes[r_idx], col_indexes[c_idx-1]:col_indexes[c_idx]] = logits_diff[c]
            # put classification value in the occluded patch
            occusion_maps_classification[row_indexes[r_idx-1]:row_indexes[r_idx], col_indexes[c_idx-1]:col_indexes[c_idx]] = patch_classification

    return occusion_maps_logits, occusion_maps_classification

def get_occlusion_maps_batched(model, image, patch_size=10, patch_value=0, debug=False, batch_size=20, channel_to_patch=None):
    '''
    Function that given a model and an image that can be infered by the model,
    creates the occlusion maps for all the classification outputs (one image per
    class).
    INPUTS
    model : tf.keras.model
        Trained model
    imges : numpy array with shape [1, width, hight, nbr_channels]
        Image that the model is able to infere on.
    patch_size : int
        Specifies how large is the size of the squared patch used to compute
        the occlusion map.
    patch_value : float
        Value to set the patch to
    channel_to_patch : int
        Specifies what channel to patch during the occlusion. If None, all channels
        are patched

    OUTPUT
    occusion_maps :  numpy arrays of shape [n_classes, width, hight]
        The rray stores the difference in logits classification between
        the patched image and the not-patched image in the pixels that are patched.

    STEPS
    - compute the indexes to get the pathes from the image based on the
    patch size.
    - for all the patches
        # set the patch to patch_value
        # compute model prediction on the patched image
        # compute difference between logits in the patched image and the original
          image
        # store the value of the difference in the occusion_map
    '''

    '''
    To be implemented: checks for the validity of the variables
    '''

    # initiate variables
    n_classes = model.output.shape[-1]
    x_shape, y_shape = image.shape[1], image.shape[2]
    occusion_maps_logits = np.zeros((n_classes, x_shape, y_shape))
    occusion_maps_classification = np.zeros((x_shape, y_shape))

     # make the last index to be the width of the image -> the last patch smaller that the actual patch size
    row_indexes = list(range(0, x_shape, patch_size))
    row_indexes.append(x_shape)
    # same for the colunms
    col_indexes = list(range(0, y_shape, patch_size))
    col_indexes.append(y_shape)

    # compute whole image prediction logits
    whole_image_logits = model(image).numpy()[0]

    # initialize accumulators
    accumulated_patches = []
    accumulated_patches_idx = []

    # compute total number of pathces and batches
    total_patches = (len(row_indexes)-1)*(len(col_indexes)-1)
    total_batches = total_patches // batch_size + total_patches % batch_size

    # loop through all the patches
    status_idx = 0
    batch_counter = 0

    for r_idx in range(1, len(row_indexes)):
        for c_idx in range(1, len(col_indexes)):
            status_idx += 1
            # accumulate images untill reached the batch size
            patched_image = np.array(image)
            if channel_to_patch is None:
                # patch all the channels
                patched_image[:, row_indexes[r_idx-1]:row_indexes[r_idx],col_indexes[c_idx-1]:col_indexes[c_idx],:] = patch_value
            else:
                # patch only a specific channel
                patched_image[:, row_indexes[r_idx-1]:row_indexes[r_idx],col_indexes[c_idx-1]:col_indexes[c_idx],channel_to_patch] = patch_value
            accumulated_patches.append(patched_image)
            # save also indexes
            accumulated_patches_idx.append([row_indexes[r_idx-1],row_indexes[r_idx],col_indexes[c_idx-1],col_indexes[c_idx]])

            # if accumulated enough
            if any([len(accumulated_patches)==batch_size, all([batch_counter==total_batches, len(accumulated_patches) == (total_patches % batch_size)])]):
                # debug
                if debug:
                    print(f'Woring on {total_patches} patches ({100*status_idx/total_patches:.02f}%) batch {batch_counter}/{total_batches} \r', end='')

                # predict batch
                accumulated_patches = np.concatenate(accumulated_patches, axis=0)
                batch_pred_logits = model(accumulated_patches).numpy()
                # compute prediction on the batch
                batch_classification = np.argmax(batch_pred_logits, axis=-1)
                # compute logits difference
                logits_diff = np.abs(whole_image_logits - batch_pred_logits)
                # put classification value in the different patches
                for i, p in enumerate(accumulated_patches_idx):
                    for c in range(n_classes):
                        occusion_maps_logits[c, p[0]:p[1], p[2]:p[3]] = logits_diff[i, c]
                    # save also the classification
                    occusion_maps_classification[p[0]:p[1], p[2]:p[3]] = batch_classification[i]
                # reset accumulators
                accumulated_patches = []
                accumulated_patches_idx = []
                batch_counter += 1

    return occusion_maps_logits, occusion_maps_classification

## print occlusion maps (TESTING)

dataset_type = 'test'
img_dataset = test_img
gt_dataset = test_gt
batch_size = 13
channel_to_patch = [None, 0,1,2,3]

# flags
fix_image = True
idx_class_of_interest = [1]
additional_axis= 3
original_image_modality = ''
class_description = {'0':'background slice',
                    '1': 'tumor slice'}

for i in range(img_dataset.shape[0]):
    print(f'Working in {dataset_type}, image {i+1}/{img_dataset.shape[0]} \r', end='')
    # get image to plot occlusion map
    image = np.array(np.expand_dims(img_dataset[i,:,:,:], axis=0))
    image_classification = np.argmax(model(image).numpy(), axis=-1)
    gt_image = np.array(np.expand_dims(gt_dataset[i,:,:,:], axis=0))

    # dict where to save the different occlusion maps
    c_map_dict = {}

    # compute occlusion map for the requested channels
    for ctp in channel_to_patch:
        occusion_maps_logits, occlusion_map_pred = get_occlusion_maps_batched(model,
                        image,
                        patch_size=5,
                        patch_value=-1,
                        debug=True,
                        batch_size=batch_size,
                        channel_to_patch=ctp)
        # save map
        c_map_dict[str(ctp)] = occusion_maps_logits
        # save anatomical image
        c_map_dict['anatomical_image'] = image

        # get modality of the anatomical image
        original_image_modality = modalities_names[0] if ctp == None else modalities_names[ctp]
        c_map_dict['anatomical_image_modality']  = original_image_modality

        # start plotting
        aus = [1 for i in range(len(idx_class_of_interest) + additional_axis)]
        aus[-1] = 0.1
        gridspec = {'width_ratios': aus,
                    'height_ratios':[1]}
        fig, axes = plt.subplots(nrows=1, ncols=len(idx_class_of_interest)+additional_axis, figsize=(20, 15))

        # put the original image in the first axes
        ax_idx = 0
        if ctp:
            original_image = np.squeeze(image[:,:,:,ctp])
        else:
            original_image = np.squeeze(image[:,:,:,0])
        if fix_image == True:
            original_image = np.rot90(original_image, k=1)
        axes[ax_idx].imshow(original_image, cmap='gray', interpolation=None)
        axes[ax_idx].set_title(f'Original image ({original_image_modality})')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

        # plot tumor annotation contour on original image
        ax_idx = 1
        tumor_annotation = np.squeeze(gt_image[:,:,:,-1])
        tumor_annotation_countour = tumor_annotation - ndimage.binary_erosion(tumor_annotation, iterations=-2).astype(tumor_annotation.dtype)
        if fix_image == True:
            tumor_annotation_countour = np.rot90(tumor_annotation, k=1)

        axes[ax_idx].imshow(original_image, cmap='gray', interpolation=None)
        axes[ax_idx].imshow(np.ma.masked_where(tumor_annotation_countour <= 0.5, tumor_annotation_countour), alpha=0.5)
        axes[ax_idx].set_title('Tumor contour image')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

        # add occlusion_map_pred
        c = class_description[str(image_classification[0])]
        ax_idx = 2
        if fix_image == True:
            occlusion_map_pred = np.rot90(occlusion_map_pred, k=1)
        im = axes[2].imshow(occlusion_map_pred, cmap='jet', interpolation=None)
        axes[ax_idx].set_title(f'Prediction change \n(full image prediction = {c})')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])
        aspect = 20
        pad_fraction = 0.5
        divider = make_axes_locatable(axes[ax_idx])
        width = axes_size.AxesY(axes[ax_idx], aspect=1./aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.colorbar(im, cax=cax)

        titles = [f'Occlusion map class {i}' for i in idx_class_of_interest]

        for idx, c_idx in enumerate(idx_class_of_interest):
            occusion_map = occusion_maps_logits[c_idx, :,:]
            if fix_image == True:
                occusion_map = np.rot90(occusion_map, k=1)
            im = axes[idx+additional_axis].imshow(occusion_map, cmap='jet', interpolation=None)
            axes[idx+additional_axis].set_title(titles[idx])
            axes[idx+additional_axis].set_xticks([])
            axes[idx+additional_axis].set_yticks([])


        # add colorbar for occlusion map
        aspect = 20
        pad_fraction = 0.5

        divider = make_axes_locatable(axes[-1])
        width = axes_size.AxesY(axes[-1], aspect=1./aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.colorbar(im, cax=cax)

        # save image
        if not None:
            save_path_occlusionMaps = os.path.join(MODEL_PATH, f'fold_{fold+1}', f'OcclusionMaps_{dataset_type}_occluding_{original_image_modality}_TEST')
        else:
            save_path_occlusionMaps = os.path.join(MODEL_PATH, f'fold_{fold+1}', f'OcclusionMaps_{dataset_type}_occluding_all_TEST')
        Path(save_path_occlusionMaps).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_path_occlusionMaps, f'OcclusionMap_slice_{i:03d}.png'), bbox_inches='tight', dpi = 100)
        plt.close(fig)
    # save dictionary for later use
    save_path_dict = os.path.join(MODEL_PATH, f'fold_{fold+1}', f'OcclusionMaps_{dataset_type}_numpy')
    Path(save_path_dict).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(save_path_dict, f'OcclusionMap_slice_{i:03d}.npy'),c_map_dict)

#
# ## print occlusion maps (VALIDATION)
#
# dataset_type = 'validation'
# img_dataset = validation_img
# gt_dataset = validation_gt
#
# # flags
# fix_image = True
# idx_class_of_interest = [1]
# additional_axis= 3
# original_image_modality = 'qT1_GD'
# class_description = {'0':'background slice',
#                     '1': 'tumor slice'}
#
# for i in range(img_dataset.shape[0]):
#     print(f'Working in {dataset_type}, image {i+1}/{img_dataset.shape[0]} \r', end='')
#     # get image to plot occlusion map
#     image = np.array(np.expand_dims(img_dataset[i,:,:,:], axis=0))
#     # to make things faster compute occlusion map only on the images that have some brain
#     # if np.sum(image==-1) >  (image.shape[1]*image.shape[2]/6):
#     #     continue
#     image_classification = np.argmax(model(image).numpy(), axis=-1)
#     gt_image = np.array(np.expand_dims(gt_dataset[i,:,:,:], axis=0))
#     # compute occlusion map
#     occusion_maps_logits, occlusion_map_pred = get_occlusion_maps_batched(model, image, patch_size=5, patch_value=-1, debug=True, batch_size=batch_size)
#
#     aus = [1 for i in range(len(idx_class_of_interest) + additional_axis)]
#     aus[-1] = 0.1
#     gridspec = {'width_ratios': aus,
#                 'height_ratios':[1]}
#     fig, axes = plt.subplots(nrows=1, ncols=len(idx_class_of_interest)+additional_axis, figsize=(20, 15))
#
#     # put the original image in the first axes
#     ax_idx = 0
#     original_image = np.squeeze(image[:,:,:,0])
#     if fix_image == True:
#         original_image = np.rot90(original_image, k=1)
#     axes[ax_idx].imshow(original_image, cmap='gray', interpolation=None)
#     axes[ax_idx].set_title(f'Original image ({original_image_modality})')
#     axes[ax_idx].set_xticks([])
#     axes[ax_idx].set_yticks([])
#
#     # plot tumor annotation contour on original image
#     ax_idx = 1
#     tumor_annotation = np.squeeze(gt_image[:,:,:,-1])
#     tumor_annotation_countour = tumor_annotation - ndimage.binary_erosion(tumor_annotation, iterations=-2).astype(tumor_annotation.dtype)
#     if fix_image == True:
#         tumor_annotation_countour = np.rot90(tumor_annotation, k=1)
#
#     axes[ax_idx].imshow(original_image, cmap='gray', interpolation=None)
#     axes[ax_idx].imshow(np.ma.masked_where(tumor_annotation_countour <= 0.5, tumor_annotation_countour), alpha=0.5)
#     axes[ax_idx].set_title('Tumor contour image')
#     axes[ax_idx].set_xticks([])
#     axes[ax_idx].set_yticks([])
#
#     # add occlusion_map_pred
#     c = class_description[str(image_classification[0])]
#     ax_idx = 2
#     if fix_image == True:
#         occlusion_map_pred = np.rot90(occlusion_map_pred, k=1)
#     im = axes[2].imshow(occlusion_map_pred, cmap='jet', interpolation=None)
#     axes[ax_idx].set_title(f'Prediction change \n(full image prediction = {c})')
#     axes[ax_idx].set_xticks([])
#     axes[ax_idx].set_yticks([])
#     aspect = 20
#     pad_fraction = 0.5
#     divider = make_axes_locatable(axes[ax_idx])
#     width = axes_size.AxesY(axes[ax_idx], aspect=1./aspect)
#     pad = axes_size.Fraction(pad_fraction, width)
#     cax = divider.append_axes("right", size=width, pad=pad)
#     plt.colorbar(im, cax=cax)
#
#     titles = [f'Occlusion map class {i}' for i in idx_class_of_interest]
#
#     for idx, c_idx in enumerate(idx_class_of_interest):
#         occusion_map = occusion_maps_logits[c_idx, :,:]
#         if fix_image == True:
#             occusion_map = np.rot90(occusion_map, k=1)
#         im = axes[idx+additional_axis].imshow(occusion_map, cmap='jet', interpolation=None)
#         axes[idx+additional_axis].set_title(titles[idx])
#         axes[idx+additional_axis].set_xticks([])
#         axes[idx+additional_axis].set_yticks([])
#
#
#     # add colorbar for occlusion map
#     aspect = 20
#     pad_fraction = 0.5
#
#     divider = make_axes_locatable(axes[-1])
#     width = axes_size.AxesY(axes[-1], aspect=1./aspect)
#     pad = axes_size.Fraction(pad_fraction, width)
#     cax = divider.append_axes("right", size=width, pad=pad)
#     plt.colorbar(im, cax=cax)
#
#     # save image
#     save_path_occlusionMaps = os.path.join(MODEL_PATH, 'fold_1', f'OcclusionMaps_{dataset_type}')
#     Path(save_path_occlusionMaps).mkdir(parents=True, exist_ok=True)
#     fig.savefig(os.path.join(save_path_occlusionMaps, f'OcclusionMap_slice_{i:03d}.png'), bbox_inches='tight', dpi = 100)
#     plt.close(fig)

# ## print occlusion maps (TRAINING)
#
# dataset_type = 'training'
# img_dataset = train_img
# gt_dataset = train_gt
#
# # flags
# fix_image = True
# idx_class_of_interest = [1]
# additional_axis= 3
# original_image_modality = 'qT1_GD'
# class_description = {'0':'background slice',
#                     '1': 'tumor slice'}
#
# for i in range(img_dataset.shape[0]):
#     print(f'Working in {dataset_type}, image {i+1}/{img_dataset.shape[0]} \r', end='')
#     # get image to plot occlusion map
#     image = np.array(np.expand_dims(img_dataset[i,:,:,:], axis=0))
#     image_classification = np.argmax(model(image).numpy(), axis=-1)
#     # to make things faster compute occlusion map only on the images that have some brain
#     # if np.sum(image==-1) >  (image.shape[1]*image.shape[2]/6):
#     #     continue
#     gt_image = np.array(np.expand_dims(gt_dataset[i,:,:,:], axis=0))
#     # compute occlusion map
#     occusion_maps_logits, occlusion_map_pred = get_occlusion_maps_batched(model, image, patch_size=5, patch_value=-1, debug=True, batch_size=batch_size)
#
#     aus = [1 for i in range(len(idx_class_of_interest) + additional_axis)]
#     aus[-1] = 0.1
#     gridspec = {'width_ratios': aus,
#                 'height_ratios':[1]}
#     fig, axes = plt.subplots(nrows=1, ncols=len(idx_class_of_interest)+additional_axis, figsize=(20, 15))
#
#     # put the original image in the first axes
#     ax_idx = 0
#     original_image = np.squeeze(image[:,:,:,0])
#     if fix_image == True:
#         original_image = np.rot90(original_image, k=1)
#     axes[ax_idx].imshow(original_image, cmap='gray', interpolation=None)
#     axes[ax_idx].set_title(f'Original image ({original_image_modality})')
#     axes[ax_idx].set_xticks([])
#     axes[ax_idx].set_yticks([])
#
#     # plot tumor annotation contour on original image
#     ax_idx = 1
#     tumor_annotation = np.squeeze(gt_image[:,:,:,-1])
#     tumor_annotation_countour = tumor_annotation - ndimage.binary_erosion(tumor_annotation, iterations=-2).astype(tumor_annotation.dtype)
#     if fix_image == True:
#         tumor_annotation_countour = np.rot90(tumor_annotation, k=1)
#
#     axes[ax_idx].imshow(original_image, cmap='gray', interpolation=None)
#     axes[ax_idx].imshow(np.ma.masked_where(tumor_annotation_countour <= 0.5, tumor_annotation_countour), alpha=0.5)
#     axes[ax_idx].set_title('Tumor contour image')
#     axes[ax_idx].set_xticks([])
#     axes[ax_idx].set_yticks([])
#
#     # add occlusion_map_pred
#     c = class_description[str(image_classification[0])]
#     ax_idx = 2
#     if fix_image == True:
#         occlusion_map_pred = np.rot90(occlusion_map_pred, k=1)
#     im = axes[2].imshow(occlusion_map_pred, cmap='jet', interpolation=None)
#     axes[ax_idx].set_title(f'Prediction change \n(full image prediction = {c})')
#     axes[ax_idx].set_xticks([])
#     axes[ax_idx].set_yticks([])
#     aspect = 20
#     pad_fraction = 0.5
#     divider = make_axes_locatable(axes[ax_idx])
#     width = axes_size.AxesY(axes[ax_idx], aspect=1./aspect)
#     pad = axes_size.Fraction(pad_fraction, width)
#     cax = divider.append_axes("right", size=width, pad=pad)
#     plt.colorbar(im, cax=cax)
#
#     titles = [f'Occlusion map class {i}' for i in idx_class_of_interest]
#
#     for idx, c_idx in enumerate(idx_class_of_interest):
#         occusion_map = occusion_maps_logits[c_idx, :,:]
#         if fix_image == True:
#             occusion_map = np.rot90(occusion_map, k=1)
#         im = axes[idx+additional_axis].imshow(occusion_map, cmap='jet', interpolation=None)
#         axes[idx+additional_axis].set_title(titles[idx])
#         axes[idx+additional_axis].set_xticks([])
#         axes[idx+additional_axis].set_yticks([])
#
#
#     # add colorbar for occlusion map
#     aspect = 20
#     pad_fraction = 0.5
#
#     divider = make_axes_locatable(axes[-1])
#     width = axes_size.AxesY(axes[-1], aspect=1./aspect)
#     pad = axes_size.Fraction(pad_fraction, width)
#     cax = divider.append_axes("right", size=width, pad=pad)
#     plt.colorbar(im, cax=cax)
#
#     # save image
#     save_path_occlusionMaps = os.path.join(MODEL_PATH, f'fold_{fold+1}, f'OcclusionMaps_{dataset_type}')
#     Path(save_path_occlusionMaps).mkdir(parents=True, exist_ok=True)
#     fig.savefig(os.path.join(save_path_occlusionMaps, f'OcclusionMap_slice_{i:03d}.png'), bbox_inches='tight', dpi = 100)
#     plt.close(fig)


# ## test loading npy
#
# test_dict = np.load("/home/iulta54/Code/P4_1-unsupervised/trained_models/Simple_CNN_5folds/qMRIGD/fold_3/OcclusionMaps_test_numpy/OcclusionMap_slice_001.npy", allow_pickle=True)










