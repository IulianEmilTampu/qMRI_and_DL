# Script that given a subject folder that has the registered volumes (using the register_volums_parallel_processing.sh), 
# performs interpolation to obtain isotropic resolution.
# Takes in input the subject forlder and the final isortopic resolution and returns the interpolated volumes saved in the Interpolated_n folder

Help()
{
    # Display Help
    echo "Bash script that given the subject folder and interpolation parameters, interpolates all the available modalities and saves them in the given destination folder."
    echo
    echo "Syntax : register_volumes -[s sr vr df]"
    echo "Required inputs:"
    echo "s    Subject folder where the MR volumes are located."
    echo "r    Spatial resolution [x,y,z]mm with z slice thikness."
    echo "v    Voxel resolution [x,y,z]voxels wit z slice thikness."
    echo "d    Destination folder"
}

while getopts hs:r:v:d: option; do
case "${option}" in
    h) # dysplay help
        Help
        exit;;
    s) subject_folder=${OPTARG};;
    r) set -f ; IFS=',' # split on space characters
       spatial_resolution=($OPTARG) ;;
    v) set -f ; IFS=',' # split on space characters
       voxel_resolution=($OPTARG) ;;
    d) destination_folder=${OPTARG};;
    \?) # Unknown input
        echo "InputError: not recognised input. Geven ${option}, expected s, r, v or d. Check help for more information"
        exit 1
esac
done

# check that subject folder exists and has the registered folder in it.
echo
echo "####################"
echo "Interpolation script"
echo "####################"
echo

# check that the subject folder exists
if ! [ -d $subject_folder ]; 
then
    echo "ValueError: the subject folder does not exist. Provide a valid one. Given ${subject_folder}"
    exit 1
else
    if ! [ -d $subject_folder/Registered ];
    then
        echo "ValueError: the given subject has no Registration folder. Run registration routine first. Given ${subject_folder}/Registered"
        exit 1
    fi      
fi

# check if destination forlder exists, if not create it# make folder where to save the registered volumes
if ! [ -d $destination_folder ]; 
then
    mkdir $destination_folder
fi
echo "Saving interpolated volumes in ${destination_folder}."
echo

# specify which modalities to interpolate
set -f ; IFS=' ' # split on space characters
mr_modalities="T1FLAIR T2 T2FLAIR T1FLAIR_GD T2_GD T2FLAIR_GD qMRI_T1 qMRI_T2 qMRI_PD qMRI_T1_GD qMRI_T2_GD qMRI_PD_GD CSF GM NON WM CSF_GD GM_GD NON_GD WM_GD"

# create reference matrix
echo "Creating trainsformation matrix"
/usr/local/fsl/bin/fslcreatehd ${voxel_resolution[0]} ${voxel_resolution[1]} ${voxel_resolution[2]} 1 ${spatial_resolution[0]} ${spatial_resolution[1]} ${spatial_resolution[2]} 1 0 0 0 16  $destination_folder/ref_matrix_tmp.nii.gz


# interpolate all the MRI modalities
for v in $mr_modalities; 
do
    if [ -f $subject_folder/Registered/$v.nii.gz ]; 
    then
        # apply transformation matrix
        echo "Interpolating ${v}..."
        /usr/local/fsl/bin/flirt -in $subject_folder/Registered/$v.nii.gz -applyxfm -init /usr/local/fsl/etc/flirtsch/ident.mat -out $destination_folder/${v} -paddingsize 0.0 -interp sinc -sincwidth 7 -sincwindow hanning -datatype float -ref $destination_folder/ref_matrix_tmp.nii.gz &
    else
        echo "Modality ${v} not found..."
    fi
done


# interpolate the annotation using NN interpolation
echo "Interpolating annotation..."
/usr/local/fsl/bin/flirt -in $subject_folder/Registered/Annotation.nii.gz -applyxfm -init /usr/local/fsl/etc/flirtsch/ident.mat -out $destination_folder/Annotation -paddingsize 0.0 -interp nearestneighbour -datatype float -ref $destination_folder/ref_matrix_tmp.nii.gz &

# remove reference matrix
wait
rm $destination_folder/ref_matrix_tmp.nii.gz

       
    








