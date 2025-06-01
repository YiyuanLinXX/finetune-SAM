#!/usr/bin/env python3
"""
inference_noprompt.py

Standalone script to load a fine-tuned SAM model (with parameters from args.json), 
perform batch inference on a directory of images, save predicted masks, compute
per-image Dice metrics (if ground truth provided), and write metrics to CSV.
"""
import os
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from models.sam import sam_model_registry
from utils.dsc import dice_coeff


def evaluate_1_slice(image_path: str, model, device: str = "cuda"):
    pil_orig = Image.open(image_path).convert('RGB')
    img_resized = transforms.Resize((1024, 1024))(pil_orig)
    img_t = transforms.ToTensor()(img_resized)
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    batch = normalize(img_t).unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb = model.image_encoder(batch)
        sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=None, masks=None)
        masks, _ = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True
        )
        pred = masks.argmax(dim=1)[0].cpu()

    return pred, pil_orig


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for SAM segmentation on a directory of images"
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing args.json and checkpoint_best.pth")
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing input image files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save output masks and metrics CSV")
    parser.add_argument("--gt_dir", default=None,
                        help="Optional directory of ground truth masks for metrics")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference: cuda or cpu")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = args.device

    ckpt_dir = Path(args.checkpoint_dir)
    args_json = ckpt_dir / 'args.json'
    if not args_json.exists():
        raise FileNotFoundError(f"args.json not found in {ckpt_dir}")
    with open(args_json, 'r') as f:
        model_args = argparse.Namespace(**json.load(f))
    model_args.dir_checkpoint = args.checkpoint_dir

    model = sam_model_registry[model_args.arch](
        model_args,
        checkpoint=str(ckpt_dir / 'checkpoint_best.pth'),
        num_classes=model_args.num_cls
    )
    model = model.to(device).eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    image_dir = Path(args.image_dir)
    for img_path in sorted(image_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            continue
        fname = img_path.stem

        pred_mask, pil_orig = evaluate_1_slice(str(img_path), model, device)

        mask_arr_orig = pred_mask.numpy().astype(np.uint8)
        if model_args.num_cls == 2:
            mask_arr_orig = (mask_arr_orig * 255).astype(np.uint8)

        pil_mask = Image.fromarray(mask_arr_orig, mode='L').resize(
            pil_orig.size, resample=Image.NEAREST
        )
        out_mask_path = out_dir / f"{fname}.png"
        pil_mask.save(str(out_mask_path))
        print(f"Saved mask: {out_mask_path}")

        if args.gt_dir:
            gt_path = Path(args.gt_dir) / f"{fname}.png"
            if gt_path.exists():
                gt = Image.open(str(gt_path)).convert('L').resize(
                    pil_orig.size, resample=Image.NEAREST
                )
                gt_arr = np.array(gt).astype(int)
                mask_arr = np.array(pil_mask).astype(int)
                entry = [fname]
                for cls in range(model_args.num_cls):
                    val = 255 if (model_args.num_cls == 2 and cls == 1) else cls
                    pred_cls = torch.tensor(mask_arr == val).float()
                    gt_cls = torch.tensor(gt_arr == cls).float()
                    dice = dice_coeff(pred_cls, gt_cls).item()
                    entry.append(dice)
                    print(f"{fname} - class {cls}: Dice={dice:.4f}")
                metrics.append(entry)
            else:
                print(f"GT not found for {fname}, skipping metrics.")

    if args.gt_dir and metrics:
        csv_path = out_dir / 'metrics.csv'
        header = ['filename'] + [f'dice_{cls}' for cls in range(model_args.num_cls)]

        n = len(metrics)
        sums = [0.0] * model_args.num_cls
        for entry in metrics:
            for i in range(model_args.num_cls):
                sums[i] += entry[1 + i]
        avg_vals = ['Average'] + [s / n for s in sums]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for entry in metrics:
                writer.writerow([entry[0]] + [f"{v:.4f}" for v in entry[1:]])
            writer.writerow(avg_vals)
        print(f"Saved metrics CSV to {csv_path}")

if __name__ == '__main__':
    main()
