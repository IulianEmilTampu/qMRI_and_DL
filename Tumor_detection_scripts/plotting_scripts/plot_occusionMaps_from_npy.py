'''
Script that given the folder where the npy files are along with the classification
configuration, plots the occlusion maps for every occluded channel independently
as well as the occusion image when all the channels are occluded.

SPEPS
1 - get paths and info about the input data used to train the model producing the
    occlusion maps
2 - load test images
3 - loop through the files
    # open .npy
    # plot
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

def plot_occlusion_map(anatomical_image, annotation_image, occlusion_image, anatomical_modality_name=None, fix_image=False, occlusion_class=None, save_path=None, draw=False):
        n_rows=1
        n_cols=4 # anatomical, anatomica with annotation, occlusion map, colorbar
        # start plotting
        aus = [1 for i in range(n_cols)]
        aus[-1] = 0.1
        gridspec = {'width_ratios': aus,
                    'height_ratios':[1]}
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 15))

        # put the anatomic image in the first axes
        ax_idx = 0
        if fix_image == True:
            anatomical_image = np.rot90(anatomical_image, k=1)
        axes[ax_idx].imshow(anatomical_image, cmap='gray', interpolation=None)
        axes[ax_idx].set_title(f'Anatomical image ({anatomical_modality_name})')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

        # plot tumor annotation contour on original image
        ax_idx = 1
        # tumor_annotation_countour = tumor_annotation - ndimage.binary_erosion(tumor_annotation, iterations=-2).astype(tumor_annotation.dtype)
        if fix_image == True:
            annotation_image = np.rot90(annotation_image, k=1)

        axes[ax_idx].imshow(anatomical_image, cmap='gray', interpolation=None)
        axes[ax_idx].imshow(np.ma.masked_where(annotation_image <= 0.5, annotation_image), alpha=0.5)
        axes[ax_idx].set_title('Annotation')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

        # add occlusion_map_
        ax_idx = 2
        if fix_image == True:
            occlusion_image = np.rot90(occlusion_image, k=1)
        im = axes[ax_idx].imshow(occlusion_image, cmap='jet', interpolation=None)
        axes[ax_idx].set_title(f'Occlusion map class {occlusion_class}' )
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

        # add colorbar for occlusion map
        aspect = 20
        pad_fraction = 0.5

        divider = make_axes_locatable(axes[-1])
        width = axes_size.AxesY(axes[-1], aspect=1./aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.colorbar(im, cax=cax)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi = 100)
            if draw:
                plt.show()
            else:
                plt.close(fig)

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

# ds = 'validation'
# for s in file_dict[ds]:
#     aus_subject = []
#     for m in file_dict[ds][s]:
#         print(f'Working on subject dataset {ds}, subject {s}, modality {m} \r', end='')
#         aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
#     validation_img.append(np.concatenate(aus_subject, axis=3))
# validation_img = np.concatenate(validation_img, axis=0)
#
# ds = 'train'
# for s in file_dict[ds]:
#     aus_subject = []
#     for m in file_dict[ds][s]:
#         print(f'Working on subject dataset {ds}, subject {s}, modality {m} \r', end='')
#         aus_subject.append(utilities.load_MR_modality([file_dict[ds][s][m]]))
#     train_img.append(np.concatenate(aus_subject, axis=3))
# train_img = np.concatenate(train_img, axis=0)

print('\n')

##


def plot_occlusion_map(anatomical_image, annotation_image, occlusion_image, anatomical_modality_name=None, fix_image=False, occlusion_class=None, save_path=None, draw=False):
        n_rows=1
        n_cols=3 # anatomical, anatomica with annotation, occlusion map
        # start plotting
        aus = [1 for i in range(n_cols)]
        aus[-1] = 0.1
        gridspec = {'width_ratios': aus,
                    'height_ratios':[1]}
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 15))

        # put the anatomic image in the first axes
        ax_idx = 0
        if fix_image == True:
            anatomical_image = np.rot90(anatomical_image, k=1)
        axes[ax_idx].imshow(anatomical_image, cmap='gray', interpolation=None)
        axes[ax_idx].set_title(f'Anatomical image ({anatomical_modality_name})')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

        # plot tumor annotation contour on original image
        ax_idx = 1
        # tumor_annotation_countour = tumor_annotation - ndimage.binary_erosion(tumor_annotation, iterations=-2).astype(tumor_annotation.dtype)
        if fix_image == True:
            annotation_image = np.rot90(annotation_image, k=1)

        axes[ax_idx].imshow(anatomical_image, cmap='gray', interpolation=None)
        axes[ax_idx].imshow(np.ma.masked_where(annotation_image <= 0.5, annotation_image), alpha=0.5)
        axes[ax_idx].set_title('Annotation')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

        # add occlusion_map_
        ax_idx = 2
        if fix_image == True:
            occlusion_image = np.rot90(occlusion_image, k=1)
        im = axes[ax_idx].imshow(occlusion_image, cmap='jet', interpolation=None)
        axes[ax_idx].set_title(f'Occlusion map class {occlusion_class}' )
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

        # add colorbar for occlusion map
        aspect = 20
        pad_fraction = 0.5

        divider = make_axes_locatable(axes[-1])
        width = axes_size.AxesY(axes[-1], aspect=1./aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.colorbar(im, cax=cax)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi = 100)
            if draw:
                plt.show()
            else:
                plt.close(fig)
## loop through the .npy files and save plots (one occluded channel at the time)
npy_path = '/home/iulta54/Code/P4_1-unsupervised/trained_models/Simple_CNN_5folds/BRATS/fold_3/OcclusionMaps_test_numpy'
npy_files = glob.glob(os.path.join(npy_path,'*.npy'))
npy_files.sort()

class_of_interest = 1
class_of_interest_labels = ['background', 'tumor']

for idx, npy_file in enumerate(npy_files):
    # load file
    ocm = np.load(npy_file, allow_pickle=True).item()

    # save image with all channles occluded. Use first anatomical as representative
    anatomical_image = np.squeeze(test_img[idx,:,:,0])
    anatomical_modality_name = modalities_names[0]

    annotation_image = np.squeeze(test_gt[idx,:,:,-1])
    fix_image=True
    occlusion_class=class_of_interest_labels[class_of_interest]

    occlusion_image = ocm['None'][class_of_interest,:,:]

    save_folder = os.path.join(MODEL_PATH, f'fold_{fold+1}', f'OcclusionMaps_test_occluding_all_TEST')
    save_path = os.path.join(save_folder, f'OcclusionMap_slice_{idx:03d}.png')
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    plot_occlusion_map(anatomical_image,
                       annotation_image,
                       occlusion_image,
                       anatomical_modality_name=None,
                       fix_image=True,
                       occlusion_class=occlusion_class,
                       save_path=save_path,
                       draw=False
                       )
    # now loop through the remaining occlusion maps
    for key, ollusion_maps in ocm.items():
        if key == 'None':
            continue
        else:
            idy = int(key)
            # save image with all channles occluded. Use first anatomical as representative
            anatomical_image = np.squeeze(test_img[idx,:,:,idy])
            anatomical_modality_name = modalities_names[idy]

            annotation_image = np.squeeze(test_gt[idx,:,:,-1])
            fix_image=True
            occlusion_class=class_of_interest_labels[class_of_interest]

            occlusion_image = ollusion_maps[class_of_interest,:,:]

            save_folder = os.path.join(MODEL_PATH, f'fold_{fold+1}', f'OcclusionMaps_test_occluding_{anatomical_modality_name}')
            save_path = os.path.join(save_folder, f'OcclusionMap_slice_{idx:03d}.png')
            Path(save_folder).mkdir(parents=True, exist_ok=True)

            plot_occlusion_map(anatomical_image,
                            annotation_image,
                            occlusion_image,
                            anatomical_modality_name=None,
                            fix_image=True,
                            occlusion_class=occlusion_class,
                            save_path=save_path,
                            draw=False
                            )

## loop through the .npy files and save plots (all occluded channels in one image)

def plot_occlusion_map_all_channels(anatomical_image, annotation_image, list_occlusion_images, anatomical_modality_name=None, list_names_occluded_channels=None, fix_image=False, occlusion_class=None, save_path=None, draw=False):
    '''
    Function that given a list of occluded images originating from occluding different
    channels, plots them in one image
    '''
    n_rows=1
    n_cols=2+len(list_occlusion_images) # anatomical, anatomica with annotation, occlusion map, colorbar
    if list_names_occluded_channels is None:
        list_names_occluded_channels = [i for i in range(len(list_occlusion_images))]

    # start plotting
    aus = [1 for i in range(n_cols)]
    aus[-1] = 0.1
    gridspec = {'width_ratios': aus,
                'height_ratios':[1]}
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 15))

    # put the anatomic image in the first axes
    ax_idx = 0
    if fix_image == True:
        anatomical_image = np.rot90(anatomical_image, k=1)
    axes[ax_idx].imshow(anatomical_image, cmap='gray', interpolation=None)
    axes[ax_idx].set_title(f'Anatomical image ({anatomical_modality_name})')
    axes[ax_idx].set_xticks([])
    axes[ax_idx].set_yticks([])

    # plot tumor annotation contour on original image
    ax_idx = 1
    # tumor_annotation_countour = tumor_annotation - ndimage.binary_erosion(tumor_annotation, iterations=-2).astype(tumor_annotation.dtype)
    if fix_image == True:
        annotation_image = np.rot90(annotation_image, k=1)

    axes[ax_idx].imshow(anatomical_image, cmap='gray', interpolation=None)
    axes[ax_idx].imshow(np.ma.masked_where(annotation_image <= 0.5, annotation_image), alpha=0.5)
    axes[ax_idx].set_title('Annotation')
    axes[ax_idx].set_xticks([])
    axes[ax_idx].set_yticks([])


    # add occlusion_maps
    for idx, ocm in enumerate(list_occlusion_images):
        ax_idx = idx + 2
        if fix_image == True:
            ocm = np.rot90(ocm, k=1)
        im = axes[ax_idx].imshow(ocm, cmap='jet', interpolation=None)
        axes[ax_idx].set_title(f'Occlusioning {list_names_occluded_channels[idx]} channel' )
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

    # # add colorbar for occlusion map
    # aspect = 20
    # pad_fraction = 0.5
    #
    # divider = make_axes_locatable(axes[-1])
    # width = axes_size.AxesY(axes[-1], aspect=1./aspect)
    # pad = axes_size.Fraction(pad_fraction, width)
    # cax = divider.append_axes("right", size=width, pad=pad)
    # plt.colorbar(im, cax=cax)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi = 100)
        if draw:
            plt.show()
        else:
            plt.close(fig)







MODEL_PATH = "/home/iulta54/Code/P4_1-unsupervised/trained_models/Simple_CNN_5folds/qMRIGD"
fold=2
npy_path = '/home/iulta54/Code/P4_1-unsupervised/trained_models/Simple_CNN_5folds/qMRIGD/fold_3/OcclusionMaps_test_numpy'
npy_files = glob.glob(os.path.join(npy_path,'*.npy'))
npy_files.sort()

class_of_interest = 1
class_of_interest_labels = ['background', 'tumor']
list_names_occluded_channels = modalities_names
list_names_occluded_channels.insert(0, 'all')

for idx, npy_file in enumerate(npy_files):
    # load file
    ocm = np.load(npy_file, allow_pickle=True).item()

    # save image with all channles occluded. Use first anatomical as representative
    anatomical_image = np.squeeze(test_img[idx,:,:,0])
    anatomical_modality_name = modalities_names[0]

    annotation_image = np.squeeze(test_gt[idx,:,:,-1])
    fix_image=True
    occlusion_class=class_of_interest_labels[class_of_interest]

    list_occlusion_images = [value[class_of_interest,:,:] for key, value in ocm.items()]


    save_folder = os.path.join(MODEL_PATH, f'fold_{fold+1}', f'OcclusionMaps_test_overall_image')
    save_path = os.path.join(save_folder, f'OcclusionMap_slice_{idx:03d}.png')
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    plot_occlusion_map_all_channels(anatomical_image,
                        annotation_image,
                        list_occlusion_images,
                        anatomical_modality_name=anatomical_modality_name,
                        list_names_occluded_channels=list_names_occluded_channels,
                        fix_image=True,
                        occlusion_class=occlusion_class,
                        save_path=save_path,
                        draw=False)













