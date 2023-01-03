#! /bin/bash

# scrip that takes the registered files from each subject and organizes them as:
# Modality_1
#  | |_subject_1
#  | |_subject_2
#  | |_subject_3
#  |
# Modality_2
#    |_subject_1
#    |_subject_2
#    |_subject_3

dataset_folder=/home/iulta54/Data/Gliom

#---------------------------------
# COPY MODALITIES FOR ALL SUBJECTS
#---------------------------------

# make sure to copy onlt the modalities that all the subjects have
#mr_modalities="T1FLAIR T2 T2FLAIR T1FLAIR_GD qMRI_T1 qMRI_T2 qMRI_PD qMRI_T1_GD qMRI_T2_GD qMRI_PD_GD CSF GM NON WM CSF_GD GM_GD NON_GD WM_GD"
# mr_modalities="T1FLAIR T2 T2FLAIR T1FLAIR_GD qMRI_T1 qMRI_T2 qMRI_PD qMRI_T1_GD qMRI_T2_GD qMRI_PD_GD"
mr_modalities="CSF GM NON WM CSF_GD GM_GD NON_GD WM_GD"

for m in $mr_modalities ; 
do
    # check that all the subjects have the modality, if not skip it
    check=1
    for SubjectNumber in 1 2 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ; 
    do
        if ! [ -f $dataset_folder/Subject${SubjectNumber}/Registered/${m}.nii.gz ] ; 
        then
            check=0
        fi
    done
    
    # if the check went thorugh, procede in copying the files
    if [ $check -eq  "1" ] ; 
    then 
        # check if the folder for the modality exists
        if ! [ -d $dataset_folder/All_${m} ] ;
        then
            # create folder
            mkdir $dataset_folder/All_${m}
        fi
        
        save_folder=${dataset_folder}/All_${m}
    
    
        for SubjectNumber in 1 2 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ; 
        do
            cp $dataset_folder/Subject${SubjectNumber}/Registered/${m}.nii.gz $save_folder/${m}_${SubjectNumber}.nii.gz
        done
     else
        echo "Modality ${m} not present in all the subjects. Not copying." 
     fi
done

#----------------------------------
# COPY ANNOTATIONS FOR ALL SUBJECTS
#----------------------------------
       
# check if the folder for the modality exists
if ! [ -d $dataset_folder/All_Annotations ] ;
then
    # create folder
    mkdir $dataset_folder/All_Annotations
fi
save_folder=$dataset_folder/All_Annotations

for SubjectNumber in 1 2 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ; 
    do
        if [ -f $dataset_folder/Subject${SubjectNumber}/Registered/Annotation.nii.gz ] ; 
        then 
            cp $dataset_folder/Subject${SubjectNumber}/Registered/Annotation.nii.gz $save_folder/Annotation_${SubjectNumber}.nii.gz
        else
            echo "Subject ${SubjectNumber} does not have the annotation."
        fi
    done  

#----------------------------------
# COPY BRAIN MASKS FOR ALL SUBJECTS
#----------------------------------
       
# check if the folder for the modality exists
if ! [ -d $dataset_folder/All_Brain_masks ] ;
then
    # create folder
    mkdir $dataset_folder/All_Brain_masks
fi
save_folder=$dataset_folder/All_Brain_masks

for SubjectNumber in 1 2 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ; 
    do
        if [ -f $dataset_folder/Subject${SubjectNumber}/Registered/brain_mask.nii.gz ] ; 
        then 
            cp $dataset_folder/Subject${SubjectNumber}/Registered/brain_mask.nii.gz $save_folder/brain_mask_${SubjectNumber}.nii.gz
        else
            echo "Subject ${SubjectNumber} does not have the brain mask."
        fi
    done 








