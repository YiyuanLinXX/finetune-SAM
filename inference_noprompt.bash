# PM_2019
# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/PM_2019/test_images' \
#   --gt_dir '/home/yl3663/finetune-SAM/datasets/PM_2019/test_gt' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/2D-SAM_vit_b_decoder_vanilla_PM_2019_noprompt_1' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/PM_2019/2D-SAM_vit_b_decoder_vanilla_PM_2019_noprompt_1/inference'\


# Canopy
python3 inference_noprompt.py \
  --image_dir '/home/yl3663/finetune-SAM/image' \
  --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/2D-SAM_vit_b_ed_adapter_Canopy_noprompt_3' \
  --output_dir '/home/yl3663/finetune-SAM/inference_results/Canopy/PM_SAM_CLIP_testset/inference'\


 