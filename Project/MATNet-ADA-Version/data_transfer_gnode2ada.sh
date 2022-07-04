#!/bin/bash
# mkdir -p /ssd_scratch/cvit/rodosingh/CVProject/
scp -r /ssd_scratch/cvit/rodosingh/CVProject/data/DAVIS-17/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/DAVIS17_Results/MATNet_epoch0/"$1" rodosingh@ada:/share3/rodosingh/CVProject/data/DAVIS-17/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/DAVIS17_Results/MATNet_epoch0/. 
echo "Copy DONE!!!"
echo "---------------------------------------------------------------------------------------------------"
