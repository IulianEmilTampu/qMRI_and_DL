# %%
'''
Code that resumes the cross-validation training in case routine was stopped.
It uses the cofig.json file to load the right data for each fold and resumes the
training for the incompleted folds.

There are a couple of scenarios to take into consideration:
- the fold has finished training -> the last model was saved so nothing to do;
- the fold has started training but did not finish -> resume training based on
    the last epoch trained. Only train for the remaining by loading the last
    seved model (the epoch to start training from is the one of the best model).
- the fold did not start training -> initialize model and train for all the epochs.

The code needs the folder where the cross validation folders for each run are
created, the path to the config.json file where the cross validation subject names
are saved (along with other information about the training) and the input
configuration modality to train on.
'''

import os
import sys
import glob
import json
import csv
import warnings
import numpy as np
import argparse
import importlib
import logging
from matplotlib import pyplot as plt
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# local imports
import utilities
import models

#%% GET VARIABLES
to_print = "    RESUMING CROSS VALIDATION TRAINING   "

print(f'\n{"-"*len(to_print)}')
print(to_print)
print(f'{"-"*len(to_print)}\n')

su_debug_flag = True

#--------------------------------------
# read the input arguments and set the base folder
#--------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(description='Run cross validation training on a combination of MRI modalities.')

    parser.add_argument('-wf','--CROSS_VALIDATION_FOLDER', required=True, help='Provide the folder where the cross validation folders are located for the model to resume training.')
    parser.add_argument('-cf','--CROSS_VALIDATION_FILE', required=True, help='Provide the fpath to the configuration file created during the trianing initialization.')
    parser.add_argument('-dc', "--DATASET_CONFIGURATION", required=True, help='Which dataset configuration to resume training.')
    parser.add_argument('-df', '--DATASE_FOLDER', required=True, help='Provide the Image Dataset Folder where the folders for each modality are located. Default equal to the one set in the first training run (from the configuration file).')
    parser.add_argument('-af', '--ANNOTATION_FOLDER', required=True, help='Provide the Annotation  Folder where annotations are located. Future versions will use the saved configuration file (Default equal to the one set in the first training run (from the configuration file)).')
    parser.add_argument('-bmf', '--BRAIN_MASK_FOLDER', required=True, help='Provide the Brain Mask Folder where annotations are located. Future versions will use the saved configuration file (Default equal to the one set in the first training run (from the configuration file)).')
    parser.add_argument('-gpu', '--GPU', default='0', help='Provide the GPU number to use for training.')
    parser.add_argument('-model_name', '--MODEL_NAME', default='MyModel', help='Name used to save the model and the scores. Future versions will use the saved configuration file (Default equal to the one set in the first training run (from the configuration file)).')
    parser.add_argument('-lr',"--LEARNING_RATE", default=0.001, help='Learning rate')
    parser.add_argument('-bs',"--BATCH_SIZE", default=3, help='Batch size. Future versions will use the saved configuration file (Default equal to the one set in the first training run (from the configuration file)).')
    parser.add_argument('-e',"--MAX_EPOCHS", default=1000, help='Batch size. Future versions will use the saved configuration file (Default equal to the one set in the first training run (from the configuration file)).')

    # seed for the choise of the training, validation and testing (useful in training, validationg and testing on the same subjects between runs)
    parser.add_argument('-rnd_seed',"--RANDOM_SEED", default=29122009, help='Seed for the random choise of training, validation and testing subjects.')

    args_dict = dict(vars(parser.parse_args()))

    # bring variable to the right format
    args_dict['DATASET_CONFIGURATION'] = int(args_dict['DATASET_CONFIGURATION'])
    args_dict['LEARNING_RATE'] = float(args_dict['LEARNING_RATE']) if args_dict['LEARNING_RATE'] else None
    args_dict['BATCH_SIZE'] = int(args_dict['BATCH_SIZE']) if args_dict['BATCH_SIZE'] else None
    args_dict['MAX_EPOCHS'] = int(args_dict['MAX_EPOCHS']) if args_dict['MAX_EPOCHS'] else None
    args_dict['RANDOM_SEED'] = int(args_dict['RANDOM_SEED']) if args_dict['RANDOM_SEED'] else None

else:
    # # # # # # # # # # # # # # DEBUG
    print("Running in debug mode.")
    args_dict = {}
    args_dict['CROSS_VALIDATION_FOLDER'] = "/home/iulta54/Code/P4-qMRI/trained_models/TEST_resume_training/Simple_2DUNet_rkv_10_fold5_lr0.001_batch3_cv_repetition_1_seed_1234/BRATS"
    args_dict['CROSS_VALIDATION_FILE'] = "/home/iulta54/Code/P4-qMRI/trained_models/TEST_resume_training/Simple_2DUNet_rkv_10_fold5_lr0.001_batch3_cv_repetition_1_seed_1234/train_val_test_subjects.json"
    args_dict['DATASET_CONFIGURATION'] = 0
    args_dict['DATASE_FOLDER'] = "/home/iulta54/Data/Gliom"
    args_dict['ANNOTATION_FOLDER'] = "/home/iulta54/Data/Gliom/All_Annotations"
    args_dict['BRAIN_MASK_FOLDER'] = "/home/iulta54/Data/Gliom/All_Brain_masks"
    # args_dict['DATASE_FOLDER'] = None
    # args_dict['ANNOTATION_FOLDER'] = None
    # args_dict['BRAIN_MASK_FOLDER'] = None
    args_dict['GPU'] = "1"
    args_dict['MODEL_NAME'] = 'TEST'
    args_dict['MAX_EPOCHS'] = 1000
    args_dict['RANDOM_SEED'] = 29122009


    args_dict['LEARNING_RATE'] = 0.001
    args_dict['BATCH_SIZE'] = 3

# open configuration file and get the subject for training, validation, test
if not os.path.isfile(args_dict['CROSS_VALIDATION_FILE']):
    raise ValueError(f'The given configuration file path is not valid. Given {args_dict["CROSS_VALIDATION_FILE"]}')
else:
    with open(args_dict['CROSS_VALIDATION_FILE']) as file:
        cross_validation_subjects = json.load(file)
        args_dict['NBR_FOLDS'] = len(cross_validation_subjects['train'])

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

#--------------------------------------
# set GPU
#--------------------------------------

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]=args_dict['GPU'];

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['AUTOGRAPH_VERBOSITY'] = "1"
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.utils import to_categorical
devices = tf.config.list_physical_devices('GPU')

if devices:
    print(f'Running training on GPU # {args_dict["GPU"]} \n')
    warnings.simplefilter(action='ignore',  category=FutureWarning)
else:
    Warning(f'ATTENTION!!! MODEL RUNNING ON CPU. Check implementation in case GPU is wanted.')

##
#-----------------------------------------------------------------------------
# Loop through the folds and get the information regarding the training status
#-----------------------------------------------------------------------------
if not os.path.isdir(args_dict["CROSS_VALIDATION_FOLDER"]):
    raise ValueError(f'The given cross validation folder is not valid. Given {args_dict["CROSS_VALIDATION_FOLDER"]}')

# get model name
args_dict['MODEL_NAME'] = os.path.basename(args_dict["CROSS_VALIDATION_FOLDER"])

folds_folders = glob.glob(os.path.join(args_dict['CROSS_VALIDATION_FOLDER'],'fold_*',''))
summary_cv_training = {}
for idx, cv_f in enumerate(folds_folders):
    # save information about the fold
    summary_cv_training[idx] = {'finished_training':False}

    # check if the last model is saved. if so, cv model has been trained fully
    if os.path.isdir(os.path.join(cv_f, 'last_model.tf')):
        # model has finished training
        summary_cv_training[idx]["finished_training"] = True
        with open(os.path.join(cv_f, 'history.json')) as file:
            history = json.load(file)
        summary_cv_training[idx]["best_model_path"] = os.path.join(cv_f, 'best_model_weights.tf')
        summary_cv_training[idx]["best_model_epoch"] = len(history["training_loss"])
        summary_cv_training[idx]["history"] = history

    else:
        # check if the history.json file is present, if so load it
        # this means that the model started to train
        if os.path.isfile(os.path.join(cv_f, 'history.json')):
            with open(os.path.join(cv_f, 'history.json')) as file:
                history = json.load(file)
            # save information
            summary_cv_training[idx]["best_model_path"] = os.path.join(cv_f, 'best_model_weights.tf')
            summary_cv_training[idx]["best_model_epoch"] = len(history["training_loss"])
            summary_cv_training[idx]["history"] = history
        else:
            # this means that the model did not start training
            summary_cv_training[idx]["best_model_path"] = None
            summary_cv_training[idx]["best_model_epoch"] = 0

# debug
summary_cv_training[1]['best_model_path'] = summary_cv_training[0]['best_model_path']
summary_cv_training[1]['best_model_epoch'] = 999
summary_cv_training[1]['history'] = summary_cv_training[0]['history']

# print summary of findings
print('Sumary fold search...')
for key, value in summary_cv_training.items():
    if value["finished_training"]:
        print(f'Fold {key+1}: Finished training (trained for {value["best_model_epoch"]} epochs)')
    else:
        print(f'Fold {key+1}: To resume training (trained for {value["best_model_epoch"]} epochs)')



##
#---------------------------------
# LOAD DATASER
#---------------------------------
print(f'\n{" "*3}Loading all the subjects in the training, validation and testing folds')
'''
Independently from the combination of modalities, the test valid and train sets
defined so that no vlomume is present in more than one set.

Steps
2 - using the number of subject to use, create indexes to identify which files
    are used for training, validation and test
3 - save the infromation about the split.
'''


combination_dict = {'0':['BRATS',['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2']],
                    '1':['qMRI',['All_qMRI_T1', 'All_qMRI_T2', 'All_qMRI_PD']],
                    '2':['qMRI_GD',['All_qMRI_T1_GD', 'All_qMRI_T2_GD', 'All_qMRI_PD_GD']],
                    '3':['BRATS-qMRI',['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2', 'All_qMRI_T1', 'All_qMRI_T2', 'All_qMRI_PD']],
                    '4':['BRATS-qMRI_GD',['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2', 'All_qMRI_T1_GD', 'All_qMRI_T2_GD', 'All_qMRI_PD_GD']],
                    '5':['qMRI-qMRI_GD',['All_qMRI_T1', 'All_qMRI_T2', 'All_qMRI_PD', 'All_qMRI_T1_GD', 'All_qMRI_T2_GD', 'All_qMRI_PD_GD']],
                    '6':[' BRATS-qMRI-qMRI_GD',['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2', 'All_qMRI_T1', 'All_qMRI_T2', 'All_qMRI_PD', 'All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2', 'All_qMRI_T1_GD', 'All_qMRI_T2_GD', 'All_qMRI_PD_GD']],
                    '7':['T1W_GD',['All_T1FLAIR_GD']],
                    '8':['Delta_Tumor_border',['All_delta_border_R1_GD', 'All_delta_border_R2_GD', 'All_delta_border_PD_GD']],
                    '9':['BRATS-Delta_Tumor_border',['All_T1FLAIR_GD', 'All_T1FLAIR', 'All_T2FLAIR', 'All_T2','All_delta_border_R1_GD', 'All_delta_border_R2_GD', 'All_delta_border_PD_GD']],
                    '10':['T1wGD_T1GD_T2GD',['All_T1FLAIR_GD', 'All_qMRI_T1_GD', 'All_qMRI_T2_GD']]
                    }

setting = combination_dict[str(args_dict['DATASET_CONFIGURATION'])][0]
modalities = combination_dict[str(args_dict['DATASET_CONFIGURATION'])][1]

# check if all modaliites are available, if not print warning and move to the next setting
check_modalities = [m for m in modalities if os.path.isdir(os.path.join(args_dict['DATASE_FOLDER'], m))==False]
if check_modalities:
    print(f'\n {"#"*10} \n ATTENTION! \n Not all modalities are available {check_modalities}! Check dataset folder. \n Moving to the next setting. \n {"#"*10}')


classes = ["Background", "Tumor"]
Nclasses = len(classes)

annotation_archive = {}
image_archive = {}

subjects_to_load = []
subjects_to_load.extend(cross_validation_subjects['test'])
subjects_to_load.extend(cross_validation_subjects['validation'][0])
subjects_to_load.extend(cross_validation_subjects['train'][0])

for ids, subj in enumerate(subjects_to_load):
    # get subject index
    subj_idx = int(subj.split('_')[-1].split('.')[0])
    print(f'Loading {ids+1}/{len(subjects_to_load)} (subject {subj_idx})...\r', end='')
    # load annotation and preprocess annotation
    aus_annotations_archive = utilities.load_3D_data([os.path.join(args_dict["ANNOTATION_FOLDER"],subj)])
    annotations = aus_annotations_archive['data_volumes'][:,:,:,:,0]
    annotations = annotations.transpose(0,3,1,2)
    annotations = annotations.reshape(annotations.shape[0]*annotations.shape[1], annotations.shape[2], annotations.shape[3])
    annotations = annotations / np.max(annotations)
    annotations = annotations.astype('int')
    # save in the archive
    annotation_archive[subj_idx] = to_categorical(annotations, Nclasses)

    # load image
    aus_images = []
    # load all modalities
    for idx, modality in enumerate(modalities):
        # load data
        file_name = os.path.join(args_dict["DATASE_FOLDER"], modality, '_'.join(modality.split('_')[1::])+'_'+str(subj_idx)+'.nii.gz')
        aus_images.append(utilities.load_MR_modality([file_name]))
    image_archive[subj_idx] = np.concatenate(aus_images, axis=3)


# %%
#---------------------------------
# TRAIN ON THE NEEDED FOLDS
#---------------------------------

# create dictionary where to save the test performance for this cross validation
summary_test = dict.fromkeys(list(summary_cv_training.keys()))
for key in summary_test.keys():
    summary_test[key] = {'best':{'Dice':0, 'Accuracy':0}, 'last':{'Dice':0, 'Accuracy':0}}
    if summary_cv_training[key]["finished_training"]:
        # load the metrics for this model
        with open(os.path.join(folds_folders[key], 'tabular_test_summary.csv')) as file:
            trained_model_performance = csv.reader(file, delimiter=',')
            next(trained_model_performance)
            for row, m_version in zip(trained_model_performance,['best','last']):
                summary_test[key][m_version]['Dice'] = float(row[6])
                summary_test[key][m_version]['Accuracy'] = float(row[7])


for cv_f, cv_f_values in summary_cv_training.items():
    if not cv_f_values['finished_training']:
        print(f'Resuming training fold {cv_f+1} (remaining epochs {args_dict["MAX_EPOCHS"] - cv_f_values["best_model_epoch"]})')
        # run training fold
        # 1 - group training training, validation and testing data
        Xtest = [image_archive[int(subj.split('_')[-1].split('.')[0])] for subj in cross_validation_subjects['test']]
        Xtest = np.concatenate(Xtest, axis=0)

        Ytest = [annotation_archive[int(subj.split('_')[-1].split('.')[0])] for subj in cross_validation_subjects['test']]
        Ytest = np.concatenate(Ytest, axis=0)

        Xtrain = [image_archive[int(subj.split('_')[-1].split('.')[0])] for subj in cross_validation_subjects['train'][cv_f]]
        Xtrain = np.concatenate(Xtrain, axis=0)

        Ytrain = [annotation_archive[int(subj.split('_')[-1].split('.')[0])] for subj in cross_validation_subjects['train'][cv_f]]
        Ytrain = np.concatenate(Ytrain, axis=0)

        Xvalid = [image_archive[int(subj.split('_')[-1].split('.')[0])] for subj in cross_validation_subjects['validation'][cv_f]]
        Xvalid = np.concatenate(Xvalid, axis=0)

        Yvalid = [annotation_archive[int(subj.split('_')[-1].split('.')[0])] for subj in cross_validation_subjects['validation'][cv_f]]
        Yvalid = np.concatenate(Yvalid, axis=0)

        # print datasets shapes
        print(f'{" "*3}{"Training dataset shape":25s}: image -> {Xtrain.shape}, mask -> {Ytrain.shape}')
        print(f'{" "*3}{"Validation dataset shape":25s}: image -> {Xvalid.shape}, mask -> {Yvalid.shape}')
        print(f'{" "*3}{"Testing dataset shape":25s}: image -> {Xtest.shape}, mask -> {Ytest.shape}')

        # build data generators
        train_gen = utilities.apply_augmentation(Xtrain, Ytrain, batch_size=args_dict['BATCH_SIZE'], seed=args_dict['RANDOM_SEED'])
        val_gen = utilities.apply_augmentation(Xvalid, Yvalid, batch_size=args_dict['BATCH_SIZE'], seed=args_dict['RANDOM_SEED'])

        # build model
        class_weights = [1, int((np.sum(Ytrain[:,:,:,0]) / np.sum(Ytrain[:,:,:,1]))**2)]
        img_size = Xtrain[0].shape
        net_aug = models.unet(img_size, Nclasses, class_weights, args_dict['MODEL_NAME'], Nfilter_start=32, batch_size=args_dict['BATCH_SIZE'], depth=4)

        # load weights of best model if available
        if cv_f_values["best_model_path"]:
            print('Loading best model weights...')
            net_aug.model.load_weights(cv_f_values["best_model_path"])
            print('Model loaded.')

        #--------------------------
        # TRAIN MODEL
        #-------------------------
        # adjust learning rate start to match
        args_dict['LEARNING_RATE'] = 0.001
        args_dict['LEARNING_RATE']  = args_dict['LEARNING_RATE'] * np.power(1-cv_f_values["best_model_epoch"]/args_dict['MAX_EPOCHS'], 0.9)

        net_aug.custum_train(train_gen, val_gen,
                        Xtrain.shape[0]/args_dict['BATCH_SIZE'],
                        Xvalid.shape[0]/args_dict['BATCH_SIZE'],
                        max_epocs=args_dict['MAX_EPOCHS'] - cv_f_values["best_model_epoch"],
                        verbose=1,
                        save_model_path=os.path.join(args_dict['CROSS_VALIDATION_FOLDER'],"fold_"+str(cv_f+1)),
                        early_stopping=True, patience=args_dict['MAX_EPOCHS'] - cv_f_values["best_model_epoch"],
                        start_learning_rate=args_dict['LEARNING_RATE'])

        #--------------------------
        # Stick together model history
        #-------------------------

        if cv_f_values["best_model_path"]:
            # save history training curves
            json_dict = OrderedDict()
            cv_f_values['history']['training_loss'].extend(net_aug.train_loss_history)
            json_dict['training_loss'] = cv_f_values['history']['training_loss']

            cv_f_values['history']['validation_loss'].extend(net_aug.val_loss_history)
            json_dict['validation_loss'] = cv_f_values['history']['validation_loss']

            cv_f_values['history']['training_accuracy'].extend(net_aug.train_accuracy_history)
            json_dict['training_accuracy'] = cv_f_values['history']['training_accuracy']

            cv_f_values['history']['validation_accuracy'].extend(net_aug.val_accuracy_history)
            json_dict['validation_accuracy'] = cv_f_values['history']['validation_accuracy']

            cv_f_values['history']['training_dice'].extend(net_aug.train_dice_history)
            json_dict['training_dice'] = cv_f_values['history']['training_dice']

            cv_f_values['history']['validation_dice'].extend(net_aug.val_dice_history)
            json_dict['validation_dice'] = cv_f_values['history']['validation_dice']

            with open(os.path.join(args_dict['CROSS_VALIDATION_FOLDER'],"fold_"+str(cv_f+1),'history.json'), 'w') as fp:
                json.dump(json_dict, fp)

            # save training curves
            utilities.plotModelPerformance_v2(json_dict['training_loss'],
                                    json_dict['training_accuracy'],
                                    json_dict['validation_loss'],
                                    json_dict['validation_accuracy'],
                                    json_dict['training_dice'],
                                    json_dict['validation_dice'],
                                    os.path.join(args_dict['CROSS_VALIDATION_FOLDER'],"fold_"+str(cv_f+1)),
                                    best_epoch =None,
                                    display=False)

        #--------------------------
        # EVALUATE LAST & BEST MODEL
        #-------------------------
        # last model
        acc, dice = net_aug.evaluate(Xtest,Ytest)
        # diceScores[dataCombination,cv_f] = dice
        summary_test[cv_f]['last']['Dice'] = dice
        summary_test[cv_f]['last']['Accuracy'] = acc

        # best model
        # load weights
        net_aug.model.load_weights(os.path.join(args_dict['CROSS_VALIDATION_FOLDER'],"fold_"+str(cv_f+1), 'best_model_weights.tf'))
        acc, dice = net_aug.evaluate(Xtest,Ytest)
        summary_test[cv_f]['best']['Dice'] = dice
        summary_test[cv_f]['best']['Accuracy'] = acc

        #-------------------------------------
        # SAVE MODEL PORFORMANCE FOR THIS FOLD
        #-------------------------------------

        # save information for this fold
        summary_file = os.path.join(args_dict['CROSS_VALIDATION_FOLDER'], f'fold_{str(cv_f+1)}', f"tabular_test_summary.csv")
        csv_file = open(summary_file, "w")
        writer = csv.writer(csv_file)
        csv_header = ['task',
                    'nbr_classes',
                    'input_configuration',
                    'model_type',
                    'model_version',
                    'fold',
                    'Dice',
                    'Accuracy',
                    ]
        writer.writerow(csv_header)
        # build rows to save in the csv file
        csv_rows = []
        for m in ['last', 'best']:
            csv_rows.append(['segmentation',
                            2,
                            setting,
                            '2D_UNet',
                            m,
                            cv_f+1,
                            summary_test[cv_f][m]['Dice'],
                            summary_test[cv_f][m]['Accuracy'],
                            ])
        writer.writerows(csv_rows)
        csv_file.close()

#----------------------------------------------------------------------
# SAVE MODEL PORFORMANCE FOR ALL THE FOLDS FOR THIS INPUT CONFIGURATION
#----------------------------------------------------------------------

summary_file = os.path.join(args_dict['CROSS_VALIDATION_FOLDER'], f"tabular_test_summary.csv")
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
csv_header = ['task',
            'nbr_classes',
            'input_configuration',
            'model_type',
            'model_version',
            'fold',
            'Dice',
            'Accuracy',
            ]
writer.writerow(csv_header)
# build rows to save in the csv file
csv_rows = []
for cv_f in range(args_dict['NBR_FOLDS']):
    for m in ['last', 'best']:
        csv_rows.append(['segmentation',
                        2,
                        setting,
                        '2D_UNet',
                        m,
                        cv_f+1,
                        summary_test[cv_f][m]['Dice'],
                        summary_test[cv_f][m]['Accuracy'],
                        ])
writer.writerows(csv_rows)
csv_file.close()










