#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from models.sam_LoRa import LoRA_Sam
# Scientific computing
import numpy as np
import os
# PyTorch
import torch
import torchvision
from torch.utils.data import DataLoader
# Visualization
from torchvision import transforms
from PIL import Image
# Others
from pathlib import Path
from tqdm import tqdm
import cfg
from argparse import Namespace
import json
from utils.dataset import Public_dataset
from utils.dsc import dice_coeff

def main(args, test_img_list):
    # Strip leading './ckpt/' or 'ckpt/' from args.dir_checkpoint
    dir_name = args.dir_checkpoint
    if dir_name.startswith('./ckpt/'):
        dir_name = dir_name[len('./ckpt/'):]
    elif dir_name.startswith('ckpt/'):
        dir_name = dir_name[len('ckpt/'):]
    save_folder = os.path.join('test_results', dir_name)
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    test_dataset = Public_dataset(
        args, args.img_folder, args.mask_folder,
        test_img_list, phase='val',
        targets=[args.targets], if_prompt=False
    )
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load model
    if args.finetune_type in ('adapter', 'vanilla'):
        sam_fine_tune = sam_model_registry[args.arch](
            args,
            checkpoint=os.path.join(args.dir_checkpoint, 'checkpoint_best.pth'),
            num_classes=args.num_cls
        )
    else:
        sam = sam_model_registry[args.arch](
            args,
            checkpoint=os.path.join(args.sam_ckpt),
            num_classes=args.num_cls
        )
        sam_fine_tune = LoRA_Sam(args, sam, r=4).to('cuda').sam
        sam_fine_tune.load_state_dict(
            torch.load(os.path.join(args.dir_checkpoint, 'checkpoint_best.pth')),
            strict=False
        )
    sam_fine_tune = sam_fine_tune.to('cuda').eval()

    # Metrics accumulators
    class_iou = torch.zeros(args.num_cls)
    cls_dsc   = torch.zeros(args.num_cls)
    eps = 1e-9

    for i, data in enumerate(tqdm(testloader, desc='Inference')):
        imgs = data['image'].to('cuda')  # [1,3,H,W]
        msks = torchvision.transforms.Resize(
            (args.out_size, args.out_size)
        )(data['mask']).to('cuda')     # [1,1,H,W]

        img_name = data['img_name'][0]

        with torch.no_grad():
            img_emb = sam_fine_tune.image_encoder(imgs)
            sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            pred_logits, _ = sam_fine_tune.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam_fine_tune.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
        pred = pred_logits.argmax(dim=1)  # [1,H,W]

        # —— Save image & mask —— #
        img_out_path  = os.path.join(save_folder, f"{img_name}_image.png")
        mask_out_path = os.path.join(save_folder, f"{img_name}_mask.png")
        os.makedirs(os.path.dirname(img_out_path), exist_ok=True)

        img_np = (imgs.cpu()[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(img_out_path)

        mask_np = pred.cpu()[0].numpy().astype(np.uint8)
        if args.num_cls > 1:
            mask_np = (mask_np * (255 // (args.num_cls - 1))).astype(np.uint8)
        Image.fromarray(mask_np).save(mask_out_path)
        # —— End save —— #

        # Compute metrics
        yhat = pred.cpu().flatten()
        y    = msks.cpu().flatten()
        for j in range(args.num_cls):
            I = ((y == j) & (yhat == j)).sum().item()
            U = ((y == j) | (yhat == j)).sum().item()
            class_iou[j] += I / (U + eps)
        for cls in range(args.num_cls):
            mask_pred_cls = (pred.cpu() == cls).float()
            mask_gt_cls   = (msks.cpu() == cls).float()
            cls_dsc[cls] += dice_coeff(mask_pred_cls, mask_gt_cls).item()

    class_iou /= (i + 1)
    cls_dsc   /= (i + 1)

    np.save(os.path.join(save_folder, 'class_iou.npy'), class_iou.numpy())
    np.save(os.path.join(save_folder, 'class_dsc.npy'), cls_dsc.numpy())

    print(args.dataset_name)
    print('class dsc:', cls_dsc)
    print('class iou:', class_iou)

if __name__ == "__main__":
    args = cfg.parse_args()
    args_path = os.path.join(args.dir_checkpoint, 'args.json')
    with open(args_path, 'r') as f:
        args = Namespace(**json.load(f))

    print('train dataset:', args.dataset_name)
    test_img_list = os.path.join(args.img_folder, args.dataset_name, 'test.csv')
    main(args, test_img_list)
