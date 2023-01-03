# script that given the folder where the subject interpolated volumes are locatate, it trims all the modalities available by a specified number of slices

Help()
{
    # Display Help
    echo "Bash script that given the folder where the interpolated volumes are located for one subject, trims all the available modalities by the given amoun."
    echo
    echo "Syntax : trim_subject -[s t]"
    echo "Required inputs:"
    echo "s     Subject folder where the MR volumes are located."
    echo "t     Amount to trim from top of the volume."
    echo "b     Amount to trim from bottom of the volume."
}

while getopts hs:t:b: option; do
case "${option}" in
    h) # dysplay help
        Help
        exit;;
    s) subject_folder=${OPTARG};;
    t) top=${OPTARG};;
    b) bottom=${OPTARG};;

    \?) # Unknown input
        echo "InputError: not recognised input. Geven ${option}, expected s, t and b. Check help for more information"
        exit 1
esac
done

# check that subject folder exists and has the registered folder in it.
echo
echo "###################"
echo "Trim subject script"
echo "###################"
echo

# check that the subject folder exists
if ! [ -d $subject_folder ]; 
then
    echo "ValueError: the subject folder does not exist. Provide a valid one. Given ${subject_folder}"
    exit 1     
fi

# specify which modalities to trim
mr_modalities="T1FLAIR T2 T2FLAIR T1FLAIR_GD T2_GD T2FLAIR_GD qMRI_T1 qMRI_T2 qMRI_PD qMRI_T1_GD qMRI_T2_GD qMRI_PD_GD CSF GM NON WM CSF_GD GM_GD NON_GD WM_GD"

# trim all the MRI modalities
for v in $mr_modalities; 
do
    if [ -f $subject_folder/$v.nii.gz ]; 
    then
        # using afni function (output, where to trim, input)
        3dZeropad -prefix $subject_folder/${v}_trimmed.nii.gz -I -$bottom -S -${top} $subject_folder/$v.nii.gz &
    fi
done
wait
# trim the annotation 
3dZeropad -prefix $subject_folder/Annotation_trimmed.nii.gz -I -${half_trim} -S -${half_trim} $subject_folder/Annotation.nii.gz 


       
    








