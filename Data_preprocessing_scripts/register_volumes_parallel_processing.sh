#!/bin/bash

# Script that given a folder subject performs the registration of the 
# different volumes and the annotation
# STEPS
# 1 - register the SE (spin echo) and qMRI volumes to the SE_T1_GD
# 2 - register the BRAVO sequence to the SE_T1_GD (save transformation matrix)
# 3 - use the saved transformation matrix to register the annotation to the SE_T1_GD (use nearest neighborhood as interpolation method).

Help()
{
    # Display Help
    echo "Bash script that given the subject folder, performs the registration of the MR volumes and annotation in it."
    echo
    echo "Syntax : register_volumes -[s]"
    echo "Required inputs:"
    echo "s     Subject folder where the MR volumes are located."
}

while getopts hs: option; do
case "${option}" in
    h) # dysplay help
        Help
        exit;;
    s) subject_folder=${OPTARG};;
    \?) # Unknown input
        echo "InputError: not recognised input. Geven ${option}, expected s (subject_folder)"
        exit 1
esac
done


echo
echo "####################"
echo "Registration script"
echo "####################"
echo

# check that the subject folder exists
if ! [ -d $subject_folder ]; then
    echo "ValueError: the subject folder does not exist. Provide a valid one. Given ${subject_folder}"
    exit 1
fi

# make folder where to save the registered volumes
if ! [ -d $subject_folder/Registered  ]; then
    mkdir $subject_folder/Registered
fi

save_folder=$subject_folder/Registered
echo "Saving registered volumes in ${save_folder}"
echo

# set up list of volumes to be registered to the SE_T1_GD
SE_reference_volume_GD="T1FLAIR_GD"
SE_reference_volume="T1FLAIR"
SE_volumes_to_register="T2 T2FLAIR"
SE_volumes_to_register_GD="T2FLAIR_GD"

# check if reference volume exists
if ! [ -f $subject_folder/$SE_reference_volume.nii ]; then
    echo "SE reference volume not found. "
    exit 1
fi

qMR_reference_volume="qMRI_T1"
qMR_reference_volume_GD="qMRI_T1_GD"
qMR_volumes_to_register="CSF GM NON qMRI_PD qMRI_T1 qMRI_T2 WM"
qMR_volumes_to_register_GD="GM_GD NON_GD qMRI_PD_GD qMRI_T2_GD qMRI_T2_GD WM_GD"

echo "Compute brain mask to be applied to all the volumes"
rm $save_folder/brain_mask.nii.gz
rm $save_folder/brain.nii.gz

/usr/local/fsl/bin/bet $subject_folder/$SE_reference_volume_GD $save_folder/brain -f 0.40 -m -R -B &
echo



# --------------------------
# 1 - register SE to SE_T1_GD
# ---------------------------
echo "Registering the reference SE volume (MRI_T1) to the reference SE_T1_GD and saving transformation matrix..."
/usr/local/fsl/bin/flirt -in $subject_folder/$SE_reference_volume.nii -ref $subject_folder/$SE_reference_volume_GD.nii -out $save_folder/${SE_reference_volume}.nii -omat $save_folder/SE_to_SE_T1_GD.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp sinc -sincwidth 7 -sincwindow hanning &

# -----------------------------
# 2 - register qMRI to SE_T1_GD
# -----------------------------
echo "Registering the reference qMRI volume (qMRI_T1) to the reference SE_T1_GD and saving transformation matrix..."
/usr/local/fsl/bin/flirt -in $subject_folder/$qMR_reference_volume.nii -ref $subject_folder/$SE_reference_volume_GD.nii -out $save_folder/${qMR_reference_volume}.nii -omat $save_folder/qMRI_to_SE_T1_GD.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp sinc -sincwidth 7 -sincwindow hanning &

# -----------------------------
# 3 - register qMRI_GD to SE_T1_GD
# -----------------------------

echo "Registering the reference qMRI volume (qMRI_T1_GD) to the reference SE_T1_GD and saving transformation matrix..."
/usr/local/fsl/bin/flirt -in $subject_folder/$qMR_reference_volume_GD.nii -ref $subject_folder/$SE_reference_volume_GD.nii -out $save_folder/${qMR_reference_volume_GD}.nii -omat $save_folder/qMRI_GD_to_SE_T1_GD.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp sinc -sincwidth 7 -sincwindow hanning &


# ----------------------------------
# 4 - register BRAVO to the SE_T1_GD
# ----------------------------------
echo "Registering the BRAVO volume to the reference SE_T1_GD and saving transformation matrix..."
/usr/local/fsl/bin/flirt -in $subject_folder/BRAVO.nii -ref $subject_folder/$SE_reference_volume_GD.nii -out $save_folder/BRAVO.nii -omat $save_folder/BRAVO_to_SE_T1_GD.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp sinc -sincwidth 7 -sincwindow hanning &

wait
# ############## MOVE VOLUMES #######
# ---------------------------------------------------
# 5 - Move SE_GD volume while applying brain mask
# ---------------------------------------------------
echo "Moving SE_GD volumes to the registered save folder while applying brain mask..."
for v in $SE_volumes_to_register_GD; do
    if [ -f $subject_folder/$v.nii ]; then
        /usr/local/fsl/bin/fslmaths $subject_folder/${v}.nii.gz -mul $save_folder/brain_mask.nii.gz $save_folder/${v}.nii.gz
    fi
done

# move also reference volume
/usr/local/fsl/bin/fslmaths $subject_folder/$SE_reference_volume_GD.nii -mul $save_folder/brain_mask.nii.gz $save_folder/${SE_reference_volume_GD}.nii.gz

# ----------------------------------------------
# 6 - Move SE volumes while applying brain mask
# ----------------------------------------------
/usr/local/fsl/bin/fslmaths $save_folder/${SE_reference_volume}.nii.gz -mul $save_folder/brain_mask.nii.gz $save_folder/${SE_reference_volume}.nii.gz

echo "Using the transformation matrix to register the remaining SE volumes"

for v in $SE_volumes_to_register; do
    if [ -f $subject_folder/$v.nii ]; then
        echo "Registering ${v}"
        /usr/local/fsl/bin/flirt -applyxfm -in $subject_folder/$v.nii -ref $subject_folder/$SE_reference_volume_GD.nii -out $save_folder/${v}.nii -init $save_folder/SE_to_SE_T1_GD.mat 
        /usr/local/fsl/bin/fslmaths $save_folder/${v}.nii.gz -mul $save_folder/brain_mask.nii.gz $save_folder/${v}.nii.gz
    fi
done

# ----------------------------------------------
# 7 - Move qMRI volumes while applying brain mask
# ----------------------------------------------
/usr/local/fsl/bin/fslmaths $save_folder/${qMR_reference_volume}.nii.gz -mul $save_folder/brain_mask.nii.gz $save_folder/${qMR_reference_volume}.nii.gz

echo "Using the transformation matrix to register the remaining qMRI volumes"

for v in $qMR_volumes_to_register; do
    if [ -f $subject_folder/$v.nii ]; then
        echo "Registering ${v}"
        /usr/local/fsl/bin/flirt -applyxfm -in $subject_folder/$v.nii -ref $subject_folder/$SE_reference_volume_GD.nii -out $save_folder/${v}.nii -init $save_folder/qMRI_to_SE_T1_GD.mat 
        /usr/local/fsl/bin/fslmaths $save_folder/${v}.nii.gz -mul $save_folder/brain_mask.nii.gz $save_folder/${v}.nii.gz
    fi
done

# ----------------------------------------------
# 8 - Move qMRI_GD volumes while applying brain mask
# ----------------------------------------------
/usr/local/fsl/bin/fslmaths $save_folder/${qMR_reference_volume_GD}.nii.gz -mul $save_folder/brain_mask.nii.gz $save_folder/${qMR_reference_volume_GD}.nii.gz

echo "Using the transformation matrix to register the remaining qMRI volumes"

for v in $qMR_volumes_to_register_GD; do
    if [ -f $subject_folder/$v.nii ]; then
        echo "Registering ${v}"
        /usr/local/fsl/bin/flirt -applyxfm -in $subject_folder/$v.nii -ref $subject_folder/$SE_reference_volume_GD.nii -out $save_folder/${v}.nii -init $save_folder/qMRI_GD_to_SE_T1_GD.mat 
        /usr/local/fsl/bin/fslmaths $save_folder/${v}.nii.gz -mul $save_folder/brain_mask.nii.gz $save_folder/${v}.nii.gz
    fi
done

# -------------------------------------------------------------------------------
# 9 - Register annotation using BRAVO to SE_T1_GD matrix (using NN interpolation)
# -------------------------------------------------------------------------------
/usr/local/fsl/bin/fslmaths $save_folder/BRAVO.nii -mul $save_folder/brain_mask.nii.gz $save_folder/BRAVO.nii

echo "Using the transformation matrix to register the annotation volume (using NN interpolation)"
/usr/local/fsl/bin/flirt -applyxfm -in $subject_folder/Annotation.nii -ref $subject_folder/$SE_reference_volume_GD.nii -out $save_folder/Annotation.nii -init $save_folder/BRAVO_to_SE_T1_GD.mat -interp nearestneighbour




