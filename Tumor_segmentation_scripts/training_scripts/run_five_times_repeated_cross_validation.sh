#!/bin/bash

Help()
{
   # Display Help
   echo "Bash script to run a five-times repeated five-fold cross validation for the segmentation model. Run this bash script within the configured python environment."
   echo
   echo "Syntax: run_training [w|d|g]"
   echo "required inputs:"
   echo "w     Working folder (where the scripts are)"
   echo "d     Dataset folder (were the data is located)"
   echo "g     GPU number on which to run training"
   echo
}

while getopts w:hd:g: option; do
case "${option}" in
   h) # display Help
       Help
       exit;;
   w) working_folder=${OPTARG};;
   d) dataset_folder=${OPTARG};;
   g) gpu=${OPTARG};;

   \?) # incorrect option
         echo "Error: Invalid input"
         exit 1
esac
done


# work on GPU
export CUDA_VISIBLE_DEVICES=$gpu

# go to the working folder
cd $working_folder

# create trained_log_file folder
if ! [ -d $working_folder/trained_models_log ]; then
   echo "Creating folder to save log."
   mkdir $working_folder/trained_models_log
fi

log_folder=$working_folder/trained_models_log


#  # ############################################################################
#  # ################################ Simple U-Net ################################
#  # ############################################################################

declare -a model_configuration=Simple_2DUNet_rkv_5

declare -a lr=0.001
declare -a batch_size=3
declare -a nbr_subjects=21
declare -a nbr_folds=5
declare -a max_epochs=1000
# data configuration: 0=BRATS, 1=qMRI, 2=qMRIGD (see run_training_routine.py for available configurations)
declare -a data_configuration=5

nbr_crossValidation=( 1 2 3 4 5 )
rndSeed=( 1234 1235 1236 1237 1238 )

# ################## DEBUG
# nbr_crossValidation=( 1 )
# rndSeed=( 1236 )

for i in ${!nbr_crossValidation[*]}; do

    repetition=${nbr_crossValidation[i]}
    seed=${rndSeed[i]}
    save_model_name="$model_configuration"_fold"$nbr_folds"_lr"$lr"_batch"$batch_size"_cv_repetition_"$repetition"_seed_"$seed"

    python3 -u run_training_routine.py --WORKING_FOLDER $working_folder --IMG_DATASET_FOLDER $dataset_folder --ANNOTATION_DATASET_FOLDER $dataset_folder/All_Annotations --BRAIN_MASK_FOLDER $dataset_folder/All_Brain_masks --MODEL_NAME $save_model_name --NBR_SUBJECTS $nbr_subjects --NBR_FOLDS $nbr_folds --LEARNING_RATE $lr --MAX_EPOCHS $max_epochs --BATCH_SIZE  $batch_size --RANDOM_SEED $seed --GPU $gpu --DATASET_CONFIGURATION $data_configuration |& tee $log_folder/$save_model_name.log


done










