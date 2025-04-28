import os
import time
import math
import csv
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import get_dataloaders
from models import get_teacher_model, get_student_model
from utils import save_checkpoint, accuracy

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s:%(message)s',
    level=logging.INFO
)

def train_one_epoch(student, teacher, projector, device, loader,
                    optimizer, epoch, total_epochs, temp, λ_kd, λ_f, λ_ce):
    student.train()
    teacher.eval()

    mse = nn.MSELoss()
    kld = nn.KLDivLoss(log_target=True)
    ce  = nn.CrossEntropyLoss()

    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for i, (x, y) in enumerate(pbar, 1):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_feats  = teacher.forward_features(x)
            t_logits = teacher(x)
        s_feats  = student.forward_features(x)
        s_logits = student(x)
        t_feats  = projector(t_feats)

        loss_kd = kld(
            F.log_softmax(s_logits/temp, 1),
            F.softmax(t_logits/temp, 1)
        ) * (temp * temp)
        loss_f  = mse(s_feats, t_feats)
        loss_ce = ce(s_logits, y)
        loss    = λ_kd*loss_kd + λ_f*loss_f + λ_ce*loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / i
        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': f"{lr:.1e}"})

    return running_loss / len(loader)

def validate(model, device, loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += accuracy(model(x), y)
    return total / len(loader)

def main():
    p = argparse.ArgumentParser(
        description="Pocket-ViT Distillation + Stats + Plots"
    )
    # core hyperparams
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--lr',           type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_pct',   type=float, default=0.1)
    p.add_argument('--temp',         type=float, default=4.0)
    p.add_argument('--λ_kd',         type=float, default=0.7)
    p.add_argument('--λ_f',          type=float, default=0.2)
    p.add_argument('--λ_ce',         type=float, default=0.1)
    p.add_argument('--device',       type=str,   default='cuda')
    p.add_argument('--output_dir',   type=str,   default='outputs')
    args = p.parse_args()

    # make dirs
    models_dir = os.path.join(args.output_dir, 'models')
    plots_dir  = os.path.join(args.output_dir, 'plots')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    logging.info(f"Using device: {args.device}")
    train_loader, val_loader, _ = get_dataloaders(args.batch_size)
    teacher = get_teacher_model(10, args.device)
    student = get_student_model(10, args.device)
    projector = nn.Linear(384, 192).to(args.device)

    optimizer = AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(step):
        warm = args.warmup_pct * args.epochs * len(train_loader)
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, args.epochs * len(train_loader) - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # trackers
    stats = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_one_epoch(
            student, teacher, projector, args.device,
            train_loader, optimizer, epoch, args.epochs,
            args.temp, args.λ_kd, args.λ_f, args.λ_ce
        )
        acc = validate(student, args.device, val_loader)
        epoch_time = time.time() - start
        lr = optimizer.param_groups[0]['lr']

        logging.info(
            f"Epoch {epoch:03d} » Loss: {loss:.4f}  "
            f"Val Acc: {acc:.4f}  Time: {epoch_time:.1f}s"
        )

        stats.append({
            'epoch': epoch,
            'loss': loss,
            'val_acc': acc,
            'lr': lr,
            'time_s': epoch_time
        })

        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            ckpt_path = os.path.join(models_dir, f'student_best_ep{epoch:03d}.pth')
            save_checkpoint(student, ckpt_path)
            logging.info(f"Saved new best model → {ckpt_path}")

    logging.info(f"Training complete. Best val acc: {best_acc:.4f}")

    # write stats CSV
    csv_path = os.path.join(args.output_dir, 'stats.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)
    logging.info(f"Wrote stats to {csv_path}")

    # plot each metric
    epochs = [s['epoch'] for s in stats]
    # 1) Loss
    plt.figure()
    plt.plot(epochs, [s['loss'] for s in stats], marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Training Loss')
    plt.title('Pocket-ViT Training Loss'); plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'training_loss.png'))
    plt.close()

    # 2) Val Acc
    plt.figure()
    plt.plot(epochs, [s['val_acc'] for s in stats], marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
    plt.title('Pocket-ViT Validation Accuracy'); plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'val_accuracy.png'))
    plt.close()

    # 3) LR schedule
    plt.figure()
    plt.plot(epochs, [s['lr'] for s in stats], marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule'); plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'lr_schedule.png'))
    plt.close()

    # 4) Epoch time
    plt.figure()
    plt.plot(epochs, [s['time_s'] for s in stats], marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Epoch Runtime (s)')
    plt.title('Epoch Time'); plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'epoch_time.png'))
    plt.close()

    logging.info(f"Saved plots to {plots_dir}")

if __name__ == '__main__':
    main()