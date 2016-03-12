#!/bin/sh

#BSUB -q short

#BSUB -J restaurant_classification
#BSUB -n 4
#BSUB -R rusage[mem=16384]

# Wall time of 48 hours
#BSUB -W 2:00

#BSUB -o "/home/an67a/yelp_kaggle/%J.out"
#BSUB -e "/home/an67a/yelp_kaggle/%J.err"

# GPU stuff
module load cudnn/v3_for_cuda_7.0
module load nvidia_driver/331.38
module load cuda/7.0.28
# Image stuff
module load libjpeg/6b
module load zlib/1.2.8
# Python stuff
module load python/2.7.9_packages/scipy/0.17.0
module load python/2.7.9_packages/theano/0.7.0
module load python/2.7.9_packages/pandas/0.17.1
python VGG_network.py













