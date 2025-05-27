
```bash
(finetuneSAM) yl3663@cairlabgpu:~/finetune-SAM$ python SingleGPU_train_finetune_noprompt.py -h
/home/yl3663/anaconda3/envs/finetuneSAM/lib/python3.11/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: '/home/yl3663/anaconda3/envs/finetuneSAM/lib/python3.11/site-packages/torchvision/image.so: undefined symbol: _ZN3c104cuda20CUDACachingAllocator9allocatorE'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
usage: SingleGPU_train_finetune_noprompt.py [-h] [-net NET] [-arch ARCH] [-baseline BASELINE] [-dataset_name DATASET_NAME] [-img_folder IMG_FOLDER] [-mask_folder MASK_FOLDER] [-train_img_list TRAIN_IMG_LIST]
                                            [-val_img_list VAL_IMG_LIST] [-targets TARGETS] [-finetune_type FINETUNE_TYPE] [-normalize_type NORMALIZE_TYPE] [-dir_checkpoint DIR_CHECKPOINT] [-num_cls NUM_CLS]
                                            [-epochs EPOCHS] [-sam_ckpt SAM_CKPT] [-type TYPE] [-vis VIS] [-reverse REVERSE] [-pretrain PRETRAIN] [-val_freq VAL_FREQ] [-gpu GPU] [-gpu_device GPU_DEVICE] [-sim_gpu SIM_GPU]
                                            [-epoch_ini EPOCH_INI] [-image_size IMAGE_SIZE] [-out_size OUT_SIZE] [-patch_size PATCH_SIZE] [-dim DIM] [-depth DEPTH] [-heads HEADS] [-mlp_dim MLP_DIM] [-w W] [-b B] [-s S]
                                            [-if_warmup IF_WARMUP] [-warmup_period WARMUP_PERIOD] [-lr LR] [-uinch UINCH] [-imp_lr IMP_LR] [-weights WEIGHTS] [-base_weights BASE_WEIGHTS] [-sim_weights SIM_WEIGHTS]
                                            [-distributed DISTRIBUTED] [-dataset DATASET] [-thd THD] [-chunk CHUNK] [-num_sample NUM_SAMPLE] [-roi_size ROI_SIZE] [-if_update_encoder IF_UPDATE_ENCODER]
                                            [-if_encoder_adapter IF_ENCODER_ADAPTER] [-encoder-adapter-depths ENCODER_ADAPTER_DEPTHS] [-if_mask_decoder_adapter IF_MASK_DECODER_ADAPTER]
                                            [-decoder_adapt_depth DECODER_ADAPT_DEPTH] [-if_encoder_lora_layer IF_ENCODER_LORA_LAYER] [-if_decoder_lora_layer IF_DECODER_LORA_LAYER] [-encoder_lora_layer ENCODER_LORA_LAYER]
                                            [-if_split_encoder_gpus IF_SPLIT_ENCODER_GPUS] [-devices DEVICES] [-gpu_fractions GPU_FRACTIONS] [-evl_chunk EVL_CHUNK]

options:
  -h, --help            show this help message and exit
  -net NET              net type
  -arch ARCH            net architecture, pick between vit_h, vit_b, vit_t
  -baseline BASELINE    baseline net type
  -dataset_name DATASET_NAME
                        the name of dataset to be finetuned
  -img_folder IMG_FOLDER
                        the folder putting images
  -mask_folder MASK_FOLDER
                        the folder putting masks
  -train_img_list TRAIN_IMG_LIST
  -val_img_list VAL_IMG_LIST
  -targets TARGETS
  -finetune_type FINETUNE_TYPE
                        normalization type, pick among vanilla,adapter,lora
  -normalize_type NORMALIZE_TYPE
                        normalization type, pick between sam or medsam
  -dir_checkpoint DIR_CHECKPOINT
                        the checkpoint folder to save final model
  -num_cls NUM_CLS      the number of output channels (need to be your target cls num +1)
  -epochs EPOCHS        the number of largest epochs to train
  -sam_ckpt SAM_CKPT    the path to the checkpoint to load
  -type TYPE            condition type:ave,rand,rand_map
  -vis VIS              visualization
  -reverse REVERSE      adversary reverse
  -pretrain PRETRAIN    adversary reverse
  -val_freq VAL_FREQ    interval between each validation
  -gpu GPU              use gpu or not
  -gpu_device GPU_DEVICE
                        use which gpu
  -sim_gpu SIM_GPU      split sim to this gpu
  -epoch_ini EPOCH_INI  start epoch
  -image_size IMAGE_SIZE
                        image_size
  -out_size OUT_SIZE    output_size
  -patch_size PATCH_SIZE
                        patch_size
  -dim DIM              dim_size
  -depth DEPTH          depth
  -heads HEADS          heads number
  -mlp_dim MLP_DIM      mlp_dim
  -w W                  number of workers for dataloader
  -b B                  batch size for dataloader
  -s S                  whether shuffle the dataset
  -if_warmup IF_WARMUP  if warm up training phase
  -warmup_period WARMUP_PERIOD
                        warm up training phase
  -lr LR                initial learning rate
  -uinch UINCH          input channel of unet
  -imp_lr IMP_LR        implicit learning rate
  -weights WEIGHTS      the weights file you want to test
  -base_weights BASE_WEIGHTS
                        the weights baseline
  -sim_weights SIM_WEIGHTS
                        the weights sim
  -distributed DISTRIBUTED
                        multi GPU ids to use
  -dataset DATASET      dataset name
  -thd THD              3d or not
  -chunk CHUNK          crop volume depth
  -num_sample NUM_SAMPLE
                        sample pos and neg
  -roi_size ROI_SIZE    resolution of roi
  -if_update_encoder IF_UPDATE_ENCODER
                        if update_image_encoder
  -if_encoder_adapter IF_ENCODER_ADAPTER
                        if add adapter to encoder
  -encoder-adapter-depths ENCODER_ADAPTER_DEPTHS
                        the depth of blocks to add adapter
  -if_mask_decoder_adapter IF_MASK_DECODER_ADAPTER
                        if add adapter to mask decoder
  -decoder_adapt_depth DECODER_ADAPT_DEPTH
                        the depth of the decoder adapter
  -if_encoder_lora_layer IF_ENCODER_LORA_LAYER
                        if add lora to encoder
  -if_decoder_lora_layer IF_DECODER_LORA_LAYER
                        if add lora to decoder
  -encoder_lora_layer ENCODER_LORA_LAYER
                        the depth of blocks to add lora, if [], it will add at each layer
  -if_split_encoder_gpus IF_SPLIT_ENCODER_GPUS
                        if split encoder to multiple gpus
  -devices DEVICES      if split encoder to multiple gpus
  -gpu_fractions GPU_FRACTIONS
                        how to split encoder to multiple gpus
  -evl_chunk EVL_CHUNK  evaluation chunk

```

```bash
  (finetuneSAM) yl3663@cairlabgpu:~/finetune-SAM$ python val_finetune_noprompt.py -h
/home/yl3663/anaconda3/envs/finetuneSAM/lib/python3.11/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: '/home/yl3663/anaconda3/envs/finetuneSAM/lib/python3.11/site-packages/torchvision/image.so: undefined symbol: _ZN3c104cuda20CUDACachingAllocator9allocatorE'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
usage: val_finetune_noprompt.py [-h] [-net NET] [-arch ARCH] [-baseline BASELINE] [-dataset_name DATASET_NAME] [-img_folder IMG_FOLDER] [-mask_folder MASK_FOLDER]
                                [-train_img_list TRAIN_IMG_LIST] [-val_img_list VAL_IMG_LIST] [-targets TARGETS] [-finetune_type FINETUNE_TYPE] [-normalize_type NORMALIZE_TYPE]
                                [-dir_checkpoint DIR_CHECKPOINT] [-num_cls NUM_CLS] [-epochs EPOCHS] [-sam_ckpt SAM_CKPT] [-type TYPE] [-vis VIS] [-reverse REVERSE]
                                [-pretrain PRETRAIN] [-val_freq VAL_FREQ] [-gpu GPU] [-gpu_device GPU_DEVICE] [-sim_gpu SIM_GPU] [-epoch_ini EPOCH_INI] [-image_size IMAGE_SIZE]
                                [-out_size OUT_SIZE] [-patch_size PATCH_SIZE] [-dim DIM] [-depth DEPTH] [-heads HEADS] [-mlp_dim MLP_DIM] [-w W] [-b B] [-s S] [-if_warmup IF_WARMUP]
                                [-warmup_period WARMUP_PERIOD] [-lr LR] [-uinch UINCH] [-imp_lr IMP_LR] [-weights WEIGHTS] [-base_weights BASE_WEIGHTS] [-sim_weights SIM_WEIGHTS]
                                [-distributed DISTRIBUTED] [-dataset DATASET] [-thd THD] [-chunk CHUNK] [-num_sample NUM_SAMPLE] [-roi_size ROI_SIZE]
                                [-if_update_encoder IF_UPDATE_ENCODER] [-if_encoder_adapter IF_ENCODER_ADAPTER] [-encoder-adapter-depths ENCODER_ADAPTER_DEPTHS]
                                [-if_mask_decoder_adapter IF_MASK_DECODER_ADAPTER] [-decoder_adapt_depth DECODER_ADAPT_DEPTH] [-if_encoder_lora_layer IF_ENCODER_LORA_LAYER]
                                [-if_decoder_lora_layer IF_DECODER_LORA_LAYER] [-encoder_lora_layer ENCODER_LORA_LAYER] [-if_split_encoder_gpus IF_SPLIT_ENCODER_GPUS]
                                [-devices DEVICES] [-gpu_fractions GPU_FRACTIONS] [-evl_chunk EVL_CHUNK]

options:
  -h, --help            show this help message and exit
  -net NET              net type
  -arch ARCH            net architecture, pick between vit_h, vit_b, vit_t
  -baseline BASELINE    baseline net type
  -dataset_name DATASET_NAME
                        the name of dataset to be finetuned
  -img_folder IMG_FOLDER
                        the folder putting images
  -mask_folder MASK_FOLDER
                        the folder putting masks
  -train_img_list TRAIN_IMG_LIST
  -val_img_list VAL_IMG_LIST
  -targets TARGETS
  -finetune_type FINETUNE_TYPE
                        normalization type, pick among vanilla,adapter,lora
  -normalize_type NORMALIZE_TYPE
                        normalization type, pick between sam or medsam
  -dir_checkpoint DIR_CHECKPOINT
                        the checkpoint folder to save final model
  -num_cls NUM_CLS      the number of output channels (need to be your target cls num +1)
  -epochs EPOCHS        the number of largest epochs to train
  -sam_ckpt SAM_CKPT    the path to the checkpoint to load
  -type TYPE            condition type:ave,rand,rand_map
  -vis VIS              visualization
  -reverse REVERSE      adversary reverse
  -pretrain PRETRAIN    adversary reverse
  -val_freq VAL_FREQ    interval between each validation
  -gpu GPU              use gpu or not
  -gpu_device GPU_DEVICE
                        use which gpu
  -sim_gpu SIM_GPU      split sim to this gpu
  -epoch_ini EPOCH_INI  start epoch
  -image_size IMAGE_SIZE
                        image_size
  -out_size OUT_SIZE    output_size
  -patch_size PATCH_SIZE
                        patch_size
  -dim DIM              dim_size
  -depth DEPTH          depth
  -heads HEADS          heads number
  -mlp_dim MLP_DIM      mlp_dim
  -w W                  number of workers for dataloader
  -b B                  batch size for dataloader
  -s S                  whether shuffle the dataset
  -if_warmup IF_WARMUP  if warm up training phase
  -warmup_period WARMUP_PERIOD
                        warm up training phase
  -lr LR                initial learning rate
  -uinch UINCH          input channel of unet
  -imp_lr IMP_LR        implicit learning rate
  -weights WEIGHTS      the weights file you want to test
  -base_weights BASE_WEIGHTS
                        the weights baseline
  -sim_weights SIM_WEIGHTS
                        the weights sim
  -distributed DISTRIBUTED
                        multi GPU ids to use
  -dataset DATASET      dataset name
  -thd THD              3d or not
  -chunk CHUNK          crop volume depth
  -num_sample NUM_SAMPLE
                        sample pos and neg
  -roi_size ROI_SIZE    resolution of roi
  -if_update_encoder IF_UPDATE_ENCODER
                        if update_image_encoder
  -if_encoder_adapter IF_ENCODER_ADAPTER
                        if add adapter to encoder
  -encoder-adapter-depths ENCODER_ADAPTER_DEPTHS
                        the depth of blocks to add adapter
  -if_mask_decoder_adapter IF_MASK_DECODER_ADAPTER
                        if add adapter to mask decoder
  -decoder_adapt_depth DECODER_ADAPT_DEPTH
                        the depth of the decoder adapter
  -if_encoder_lora_layer IF_ENCODER_LORA_LAYER
                        if add lora to encoder
  -if_decoder_lora_layer IF_DECODER_LORA_LAYER
                        if add lora to decoder
  -encoder_lora_layer ENCODER_LORA_LAYER
                        the depth of blocks to add lora, if [], it will add at each layer
  -if_split_encoder_gpus IF_SPLIT_ENCODER_GPUS
                        if split encoder to multiple gpus
  -devices DEVICES      if split encoder to multiple gpus
  -gpu_fractions GPU_FRACTIONS
                        how to split encoder to multiple gpus
  -evl_chunk EVL_CHUNK  evaluation chunk
  ```