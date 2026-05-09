# train.py
import os
import time
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, RandomSampler

from Model import M2CDViT
from tool.Dataset import RSDataset
from tool.Losses import BCELoss, DiceLoss, ContrastiveChangeLoss
from tool.Schedulers import get_scheduler
from tool.Optimizers import get_optimizer
from Val import evaluate
from Config import CONFIG, DATASET_CONFIG

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Train MTCNet")
    parser.add_argument('--dataset', type=str, required=True, choices=['M2Change-CZ1', 'M2Change-CZ2'],
                        help="Choose dataset to train on: M2Change-CZ1 or M2Change-CZ2")
    return parser.parse_args()

def log_metrics(oa, miou, macc, mf1, mrecall, log_file):
    with open(log_file, 'a') as f:
        f.write(f"Current OA: {oa:.3f}, mIoU: {miou:.3f}, mAcc: {macc:.3f}, mF1: {mf1:.3f}, mRecall: {mrecall:.3f}\n")

def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_cudnn() -> None:
    cudnn.benchmark = True
    cudnn.deterministic = False

def main():
    args = parse_args()
    cfg = CONFIG
    ds_cfg = DATASET_CONFIG[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init save paths
    save_dir = Path(ds_cfg["log_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "log.txt"

    fix_seeds(cfg["seed"])
    setup_cudnn()

    # Model init
    model = M2CDViT(n_times=cfg["n_times"]).to(device)
    if cfg["use_dp"]:
        model = nn.DataParallel(model)

    # Loss functions
    loss_1 = BCELoss(ignore_label=cfg["ignore_label"]).to(device)
    loss_2 = DiceLoss(ignore_label=cfg["ignore_label"]).to(device)
    loss_c = ContrastiveChangeLoss().to(device)
    loss_fn = [loss_1, loss_2]  

    # Dataloaders
    trainset = RSDataset(ds_cfg["dataset_root"], "train", timestep=cfg["n_times"])
    valset = RSDataset(ds_cfg["dataset_root"], "test", timestep=cfg["n_times"])

    trainloader = DataLoader(trainset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                             pin_memory=True, drop_last=True, sampler=RandomSampler(trainset))
    valloader = DataLoader(valset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=False)

    # Optimizer & Scheduler
    optimizer = get_optimizer(model, cfg["optimizer_name"], cfg["learning_rate"], cfg["weight_decay"])
    iters_per_epoch = len(trainloader)
    scheduler = get_scheduler(cfg["scheduler_name"], optimizer, cfg["epochs"] * iters_per_epoch, 
                              cfg["scheduler_power"], iters_per_epoch * cfg["warmup_iters"], cfg["warmup_ratio"])

    best_miou = 0.0
    start_time = time.time()

    # Training loop
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{cfg['epochs']}]")

        for i, (hr, s1, gt) in pbar:
            hr, s1, gt = hr.to(device).float(), s1.to(device).float(), gt.to(device).long()
            optimizer.zero_grad()

            logits, aux, w = model(hr, s1) 

            loss_main = sum(fn(logits, gt) for fn in loss_fn)
            loss_aux = sum(fn(aux, gt) for fn in loss_fn)
            loss_clip = (loss_c(w[0], gt) + loss_c(w[1], gt) + loss_c(w[2], gt) + loss_c(w[3], gt)) / 4.0 
            
            loss = loss_main + 0.1 * loss_clip + 0.1 * loss_aux

            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()

            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_description(
                f"Epoch: [{epoch+1}/{cfg['epochs']}] Iter: [{i+1}/{iters_per_epoch}] "
                f"LR: {current_lr:.6f} Total: {total_loss/(i+1):.4f} "
                f"Main: {loss_main.item():.4f} Aux: {loss_aux.item():.4f} Clip: {loss_clip.item():.4f}"
            )

        # Validation phase
        if (epoch + 1) % cfg["eval_interval"] == 0 or (epoch + 1) == cfg["epochs"]:
            acc, macc, f1, mf1, ious, miou, oa, recall, mrecall, kappa = evaluate(model, valloader, device)
            log_metrics(oa, miou, macc, mf1, mrecall, log_file)

            # Print metrics
            table = {
                'Class': list(trainset.CLASSES) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Precision': acc + [macc],
                'Recall': recall + [mrecall]
            }
            print(tabulate(table, headers='keys'))
            print(f"Overall Accuracy (OA): {oa:.3f}, Mean IoU: {miou:.3f}, Mean F1: {mf1:.3f}")

            # Save best model
            if miou > best_miou and all(iou != 0 for iou in ious):
                best_miou = miou
                save_path = save_dir / f"{cfg['model_name']}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

    elapsed = time.gmtime(time.time() - start_time)
    print(tabulate([
        ['Best mIoU', f"{best_miou:.2f}"],
        ['Training Time', time.strftime("%H:%M:%S", elapsed)]
    ]))

if __name__ == '__main__':
    main()