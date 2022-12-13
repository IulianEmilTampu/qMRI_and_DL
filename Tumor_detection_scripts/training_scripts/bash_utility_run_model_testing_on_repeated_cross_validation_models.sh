#!/bin/bash

Help()
{
   # Display Help
   echo "Bash utility that runts the model testing for the models trained thru repeated cross validation"
   echo
   echo "Syntax: run_training [w|d|g]"
   echo "required inputs:"
   echo "w     Working folder (where the scripts are)."
   echo "m     Models folder (where the folders for each repetition are)."
   echo "d     Dataset folder (were the data is located)."
   echo "g     GPU number on which to run testing."
   echo "c     Name of the conda environment where the python interpreter is."
   echo
}

while getopts w:hd:g:c: option; do
case "${option}" in
   h) # display Help
       Help
       exit;;
   w) working_folder=${OPTARG};;
   m) models_folder=${OPTARG};;
   d) dataset_folder=${OPTARG};;
   g) gpu=${OPTARG};;
   c) conda_env_name=${OPTARG};;

   \?) # incorrect option
         echo "Error: Invalid input"
         exit 1
esac
done


# make sure to have the right conda environment open when running the script
# activate conda environment
eval "$(conda shell.bash hook)"
conda activate $conda_env_name

# work on GPU 0
export CUDA_VISIBLE_DEVICES=$gpu

# go to the working folder
cd $working_folder

#  # ############################################################################
#  # ################################ Start testing #############################
#  # ############################################################################

#input_config=( BRATS qMRI )
#input_config=( qMRIGD qMRI-qMRIGD )
input_config=( BRATS )

for d in $models_folder/*/ ; do
    for ic in ${!input_config[*]} ; do
      model_path=$d${input_config[ic]}
      
      python3 test_routine.py --WORKING_FOLDER $working_folder --IMG_DATASET_FOLDER $dataset_folder --ANNOTATION_DATASET_FOLDER $dataset_folder/All_Annotations --BRAIN_MASK_FOLDER $dataset_folder/All_Brain_masks --MODEL_PATH $model_path --TRAIN_VALIDATION_TEST_FILE ${d}train_val_test_subjects.json --GPU_NBR $gpu
    done
done
    








