# val.py
import os
import argparse
import warnings
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from Model import M2CDViT
from tool.Dataset import RSDataset
from tool.Metrics import Metrics
from Config import CONFIG, DATASET_CONFIG

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MCTNet")
    parser.add_argument('--dataset', type=str, required=True, choices=['M2Change-CZ1', 'M2Change-CZ2'],
                        help="Choose dataset to evaluate: M2Change-CZ1 or M2Change-CZ2")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Optional: Override the model path defined in config.py")
    parser.add_argument('--save_dir', type=str, default=None,
                        help="Optional: Override the save directory for output images")
    return parser.parse_args()

def setup_cudnn() -> None:
    cudnn.benchmark = True
    cudnn.deterministic = False

@torch.no_grad()
def evaluate(model, dataloader, device, mode='train', save_dir=None):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(2, 255, device)

    vis_dir = os.path.join(save_dir, "vis") if save_dir else None
    err_dir = os.path.join(save_dir, "err") if save_dir else None

    if mode != 'train' and save_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(err_dir, exist_ok=True)
    
    for data_list in tqdm(dataloader):
        if mode == 'train':
            hr, s1, gt = data_list
        else:
            hr, s1, gt, index_name = data_list

        hr, s1, gt = hr.to(device), s1.to(device), gt.to(device)
        preds, aux, w = model(hr, s1)
        
        preds = preds.softmax(dim=1)
        pred_classes = preds.argmax(dim=1)  # [B, H, W]

        # Save inference results
        if mode != 'train' and save_dir is not None:
            for i in range(pred_classes.shape[0]):
                pred_img = pred_classes[i].cpu().numpy().astype(np.uint8)
                gt_img = gt[i].cpu().numpy().astype(np.uint8)
                base_name = os.path.splitext(os.path.basename(index_name[i]))[0]

                # Save prediction mask
                vis_path = os.path.join(vis_dir, f"{base_name}.png")
                Image.fromarray(pred_img * 255).save(vis_path)

                # Generate error map
                pred_bin = (pred_img > 0).astype(np.uint8)
                gt_bin = (gt_img > 0).astype(np.uint8)

                tp = (pred_bin == 1) & (gt_bin == 1)
                tn = (pred_bin == 0) & (gt_bin == 0)
                fp = (pred_bin == 1) & (gt_bin == 0)
                fn = (pred_bin == 0) & (gt_bin == 1)

                err_vis = np.zeros((pred_img.shape[0], pred_img.shape[1], 3), dtype=np.uint8)
                err_vis[tp] = [255, 255, 255]   # White
                err_vis[fp] = [0, 0, 255]       # Red
                err_vis[fn] = [0, 255, 0]       # Green
                err_vis[tn] = [0, 0, 0]         # Black

                # Save error map
                err_path = os.path.join(err_dir, f"{base_name}.png")
                cv2.imwrite(err_path, err_vis)

        metrics.update(preds, gt)

    return (
        *metrics.compute_pixel_acc(), 
        *metrics.compute_f1(), 
        *metrics.compute_iou(), 
        metrics.compute_oa(), 
        *metrics.compute_recall(), 
        metrics.compute_kappa()
    )


def main():
    args = parse_args()
    cfg = CONFIG
    ds_cfg = DATASET_CONFIG[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Apply overrides if provided via CLI, otherwise use config.py defaults
    model_path = args.model_path if args.model_path else ds_cfg["val_model_path"]
    save_dir = args.save_dir if args.save_dir else ds_cfg["val_save_dir"]

    # Dataloader
    dataset = RSDataset(ds_cfg["dataset_root"], 'test', timestep=cfg["n_times"], mode="inf")
    dataloader = DataLoader(dataset, batch_size=cfg["test_batch_size"], num_workers=cfg["num_workers"], pin_memory=True)

    # Model initialization
    model = M2CDViT(n_times=cfg["n_times"]).to(device)
    if cfg["use_dp"]:
        model = nn.DataParallel(model)

    print(f"Loading checkpoint from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model = model.to(device)
    
    # Evaluation
    acc, macc, f1, mf1, ious, miou, oa, recall, mrecall, kappa = evaluate(
        model, dataloader, device, mode="test", save_dir=save_dir
    )

    # Output metrics
    table = {
        'Class': list(dataset.CLASSES) + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Precision': acc + [macc],
        'Recall': recall + [mrecall]
    }

    print(tabulate(table, headers='keys'))
    print(f"Overall Accuracy (OA): {oa:.3f}")
    print(f"Mean IoU: {miou:.3f}")
    print(f"Mean F1 Score: {mf1:.3f}")
    print(f"Kappa: {kappa:.3f}")


if __name__ == '__main__':
    setup_cudnn()
    main()