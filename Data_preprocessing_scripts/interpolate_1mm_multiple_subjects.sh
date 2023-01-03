#!/bin/bash

# Script that uses the register_volume.sh script to register multiple subjects
# s 1 and 2 512 512 158 1 0.4297 0.4297 1 1 0 0 0 16
# rest 512 512 144 1 0.4297 0.4297 1 1 0 0 0 16
destination_folder=Interpolated_1mm

#for SubjectNumber in 1 2 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23; 
for SubjectNumber in 1 2;
do
    # interpolation
    if [ ${SubjectNumber} -eq "1" ] || [ ${SubjectNumber} -eq "2" ] ;
    then
        # proccess 1 and 2 differently given the difference slice thickness
        ./interpolate_subject.sh -s ./Subject${SubjectNumber} -r 0.4297,0.4297,1 -v 512,512,158,1 -d /home/iulta54/Data/Gliom/Subject${SubjectNumber}/$destination_folder
    else
        ./interpolate_subject.sh -s ./Subject${SubjectNumber} -r 0.4297,0.4297,1 -v 512,512,144,1 -d /home/iulta54/Data/Gliom/Subject${SubjectNumber}/$destination_folder
    fi
    
    # fix the first and second subject
    if [ ${SubjectNumber} -eq "1" ] || [ ${SubjectNumber} -eq "2" ] ;
    then
        # run trim script 
        ./trim_subject.sh -s ./Subject${SubjectNumber}/$destination_folder -t 7 -b 7
    fi
done
