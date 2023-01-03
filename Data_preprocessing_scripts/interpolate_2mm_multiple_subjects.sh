#!/bin/bash

# Script that uses the register_volume.sh script to register multiple subjects
# s 1 and 2 512 512 79 1 0.4297 0.4297 1 1 0 0 0 16
# rest 512 512 72 1 0.4297 0.4297 1 1 0 0 0 16
destination_folder=Interpolated_2mm

for SubjectNumber in 1 2 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23; 
do
    # interpolation
    if [ ${SubjectNumber} -eq "1" ] || [ ${SubjectNumber} -eq "2" ] ;
    then
        # proccess 1 and 2 differently given the difference slice thickness
        /home/iulta54/Code/P4-qMRI/interpolate_subject.sh -s /home/iulta54/Data/Gliom/Subject${SubjectNumber} -r 0.4297,0.4297,1 -v 512,512,158,1 -d /home/iulta54/Data/Gliom/Subject${SubjectNumber}/$destination_folder
    else
        /home/iulta54/Code/P4-qMRI/interpolate_subject.sh -s /home/iulta54/Data/Gliom/Subject${SubjectNumber} -r 0.4297,0.4297,1 -v 512,512,144,1 -d /home/iulta54/Data/Gliom/Subject${SubjectNumber}/$destination_folder
    fi
    
    # fix the first and second subject
    if [ ${SubjectNumber} -eq "1" ] || [ ${SubjectNumber} -eq "2" ] ;
    then
        # run trim script 
        /home/iulta54/Code/P4-qMRI/trim_subject.sh -s /home/iulta54/Data/Gliom/Subject${SubjectNumber}/$destination_folder -t 3 -b 4
    fi
done
