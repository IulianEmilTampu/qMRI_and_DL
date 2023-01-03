#!/bin/bash

# Script that uses the register_volume.sh script to register multiple subjects

for SubjectNumber in 3 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
    /home/iulta54/Code/P4-qMRI/register_volumes_parallel_processing.sh -s /home/iulta54/Data/Gliom/Subject${SubjectNumber}
done



