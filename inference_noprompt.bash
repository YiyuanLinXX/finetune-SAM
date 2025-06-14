python3 inference_noprompt.py \
  --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/2D-SAM_vit_h_d_lora_Califlower_thermal_noprompt' \
  --output_dir '/home/yl3663/finetune-SAM/inference_results/Califlower_thermal/2D-SAM_vit_h_d_lora_Califlower_thermal_noprompt'\
  --image_dir '/home/yl3663/finetune-SAM/datasets/Califlower_thermal/test_images' \
  --gt_dir '/home/yl3663/finetune-SAM/datasets/Califlower_thermal/test_gt' \

python3 inference_noprompt.py \
  --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/2D-SAM_vit_h_d_adapter_Califlower_thermal_noprompt' \
  --output_dir '/home/yl3663/finetune-SAM/inference_results/Califlower_thermal/2D-SAM_vit_h_d_adapter_Califlower_thermal_noprompt'\
  --image_dir '/home/yl3663/finetune-SAM/datasets/Califlower_thermal/test_images' \
  --gt_dir '/home/yl3663/finetune-SAM/datasets/Califlower_thermal/test_gt' \

 python3 inference_noprompt.py \
  --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/2D-SAM_vit_h_ed_lora_Califlower_thermal_noprompt' \
  --output_dir '/home/yl3663/finetune-SAM/inference_results/Califlower_thermal/2D-SAM_vit_h_ed_lora_Califlower_thermal_noprompt'\
  --image_dir '/home/yl3663/finetune-SAM/datasets/Califlower_thermal/test_images' \
  --gt_dir '/home/yl3663/finetune-SAM/datasets/Califlower_thermal/test_gt' \


# # PM_2019
# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/2D-SAM_vit_h_d_adapter_PM_2019_noprompt' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_h_d_adapter_PM_2019_noprompt'\
#   # --gt_dir '/home/yl3663/finetune-SAM/datasets/PM_2019/test_gt' \

# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/2D-SAM_vit_b_ed_adapter_PM_2019_noprompt' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_b_ed_adapter_PM_2019_noprompt'\

# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/2D-SAM_vit_b_ed_vanilla_PM_2019_noprompt' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_b_ed_vanilla_PM_2019_noprompt'\

# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/2D-SAM_vit_h_ed_vanilla_PM_2019_noprompt' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_h_ed_vanilla_PM_2019_noprompt'\

# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/2D-SAM_vit_b_d_vanilla_PM_2019_noprompt' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_b_d_vanilla_PM_2019_noprompt'\




# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/old/2D-SAM_vit_b_decoder_adapter_PM_2019_noprompt_3' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_b_decoder_adapter_PM_2019_noprompt_3'\

# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/old/2D-SAM_vit_h_decoder_adapter_PM_2019_noprompt_3' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_h_decoder_adapter_PM_2019_noprompt_3'\

# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/old/2D-SAM_vit_b_decoder_vanilla_PM_2019_noprompt_2' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_b_decoder_vanilla_PM_2019_noprompt_2'\

# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/old/2D-SAM_vit_b_decoder_vanilla_PM_2019_noprompt_1' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_b_decoder_vanilla_PM_2019_noprompt_1'\  

# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/datasets/CA_2023/images' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/PM_2019/old/2D-SAM_vit_h_decoder_vanilla_PM_2019_noprompt_2' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Parlier_test/2D-SAM_vit_h_decoder_vanilla_PM_2019_noprompt_2'\  
# # Canopy
# python3 inference_noprompt.py \
#   --image_dir '/home/yl3663/finetune-SAM/image' \
#   --checkpoint_dir '/home/yl3663/finetune-SAM/ckpt/2D-SAM_vit_b_ed_adapter_Canopy_noprompt_3' \
#   --output_dir '/home/yl3663/finetune-SAM/inference_results/Canopy/PM_SAM_CLIP_testset/inference'\


 