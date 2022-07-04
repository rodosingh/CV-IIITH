#!/bin/bash
mkdir -p /ssd_scratch/cvit/rodosingh/CVProject/
mkdir ckpt
#scp -r rodosingh@ada:/share3/rodosingh/CVProject/data/ /ssd_scratch/cvit/rodosingh/CVProject/.
scp -r rodosingh@ada:/share3/rodosingh/CVProject/pre-trained-models/ ckpt/.
echo "Copy DONE!!!"
echo "---------------------------------------------------------------------------------------------------"
