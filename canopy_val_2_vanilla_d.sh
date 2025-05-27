#!/bin/bash

# Vanilla fine-tuning
# Update only Mask Decoder

# Set CUDA device
export CUDA_VISIBLE_DEVICES="5"

# Define variables
arch="vit_h"  # Change this value as needed
# arch="vit_b"  # Change this value as needed

finetune_type="vanilla"
dataset_name="Canopy"
targets='combine_all' # make it as binary segmentation 'multi_all' for multi cls segmentation
# Construct train and validation image list paths
img_folder="./datasets"  # Assuming this is the folder where images are stored

# Construct the checkpoint directory argument
dir_checkpoint="./ckpt/2D-SAM_${arch}_decoder_${finetune_type}_${dataset_name}_noprompt_2"

# Run the Python script
python pm_val_finetune_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_update_encoder True \
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint"