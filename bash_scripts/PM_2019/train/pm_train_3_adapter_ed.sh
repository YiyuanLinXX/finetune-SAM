#!/bin/bash

# fine-tuning using Adapter blocks
# Update both Encoder and Mask Decoder

# Set CUDA device
export CUDA_VISIBLE_DEVICES="3"

# Define variables
arch="vit_h"  # Change this value as needed
finetune_type="adapter"
dataset_name="PM_2019"
targets='combine_all' # make it as binary segmentation 'multi_all' for multi cls segmentation
# Construct train and validation image list paths
img_folder="./datasets"  # Assuming this is the folder where images are stored
train_img_list="${img_folder}/${dataset_name}/train.csv"
val_img_list="${img_folder}/${dataset_name}/val.csv"


# Construct the checkpoint directory argument
dir_checkpoint="./ckpt/2D-SAM_${arch}_decoder_${finetune_type}_${dataset_name}_noprompt_3"

# Run the Python script
python SingleGPU_train_finetune_noprompt.py \
    -if_warmup True \
    -b 1 \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_mask_decoder_adapter True \
    -if_update_encoder True \
    -if_encoder_adapter True \
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -sam_ckpt "./weights/sam_vit_h_4b8939.pth" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" \
    -val_img_list "$val_img_list" \
    -epochs 1000
