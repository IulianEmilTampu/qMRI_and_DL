#!/bin/bash

Help()
{
   # Display Help
   echo "Bash script to run a ten-times repeated five-fold cross validation"
   echo
   echo "Syntax: run_training [w|d|g]"
   echo "required inputs:"
   echo "w     Working folder (where the scripts are)."
   echo "d     Dataset folder (were the data is located)."
   echo "g     GPU number on which to run training."
   echo "c     Name of the conda environment where the python interpreter is."
   echo
}

while getopts w:hd:g:c: option; do
case "${option}" in
   h) # display Help
       Help
       exit;;
   w) working_folder=${OPTARG};;
   d) dataset_folder=${OPTARG};;
   c) conda_env_name=${OPTARG};;
   g) gpu=${OPTARG};;

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

# create trained_log_file folder
if ! [ -d $working_folder/trained_models_log ]; then
   echo "Creating folder to save log."
   mkdir $working_folder/trained_models_log
fi

log_folder=$working_folder/trained_models_log


#  # ############################################################################
#  # ################################ Simple CNN ################################
#  # ############################################################################

declare -a dataset_configuration=2
declare -a model_configuration=Simple_CNN_rkv_10

declare -a lr=0.00001
declare -a batchSize=16
declare -a nFolds=5
declare -a maxEpochs=300

nbr_crossValidation=( 1 2 3 4 5 6 7 8 9 10 )
rndSeed=( 1234 1235 1236 1237 1238 1239 1240 1241 1242 1243 )

# ################## DEBUG
# nbr_crossValidation=( 1 )
# rndSeed=( 1234 )

for i in ${!nbr_crossValidation[*]}; do

   repetition=${nbr_crossValidation[i]}
   seed=${rndSeed[i]}
   save_model_name="$model_configuration"_fold"$nFolds"_lr"$lr"_batch"$batchSize"_cv_repetition_"$repetition"_seed_"$seed"
    
   python3 -u run_training_routine.py --WORKING_FOLDER $working_folder --IMG_DATASET_FOLDER $dataset_folder --ANNOTATION_DATASET_FOLDER $dataset_folder/All_Annotations --BRAIN_MASK_FOLDER $dataset_folder/All_Brain_masks --MODEL_NAME $save_model_name --NBR_SUBJECTS 21 --NBR_FOLDS $nFolds --BATCH_SIZE $batchSize --LEARNING_RATE $lr --MAX_EPOCHS $maxEpochs --DATASET_CONFIGURATION $dataset_configuration --GPU_NBR $gpu --RANDOM_SEED_NUMBER $seed |& tee $log_folder/$save_model_name.log

done










