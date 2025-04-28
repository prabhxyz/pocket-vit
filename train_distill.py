# full distillation loop w/ logit + feature + CE + plotting
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from data import get_dataloaders
from models import get_teacher_model, get_student_model
from utils import accuracy, save_checkpoint
import matplotlib.pyplot as plt  # for plotting
import argparse

def train_one_epoch(student, teacher, projector, device, loader,
                    optimizer, epoch, total_epochs, temp, λ_kd, λ_f, λ_ce):
    student.train(); teacher.eval()
    mse = nn.MSELoss(); kld = nn.KLDivLoss(log_target=True); ce = nn.CrossEntropyLoss()
    running = 0.0
    for i, (x,y) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            t_feats = teacher.forward_features(x)
            t_logits = teacher(x)
        s_feats = student.forward_features(x)
        s_logits = student(x)
        t_feats = projector(t_feats)

        loss_kd = kld(
            F.log_softmax(s_logits/temp,1),
            F.softmax(t_logits/temp,1)
        ) * (temp*temp)
        loss_f  = mse(s_feats, t_feats)
        loss_ce = ce(s_logits,y)
        loss = λ_kd*loss_kd + λ_f*loss_f + λ_ce*loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
        if i % 100 == 0:
            print(f'Epoch[{epoch}/{total_epochs}] Step[{i}/{len(loader)}] '
                  f'Loss:{running/(i+1):.4f}')
    return running / len(loader)

def validate(model, device, loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            total += accuracy(model(x), y)
    return total / len(loader)

def main():
    p = argparse.ArgumentParser()
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
    args = p.parse_args()

    train_loader, val_loader, _ = get_dataloaders(args.batch_size)
    teacher = get_teacher_model(10, args.device)
    student = get_student_model(10, args.device)
    projector = nn.Linear(384,192).to(args.device)

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

    train_losses = []
    val_accs     = []
    best_acc     = 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            student, teacher, projector, args.device,
            train_loader, optimizer, epoch, args.epochs,
            args.temp, args.λ_kd, args.λ_f, args.λ_ce
        )
        train_losses.append(loss)

        acc = validate(student, args.device, val_loader)
        val_accs.append(acc)
        print(f'>>> Epoch {epoch} Val Acc: {acc:.4f}')

        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(student, 'student_best.pth')

    print('Training complete. Best val acc:', best_acc)

    # --- plotting ---
    epochs = list(range(1, args.epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Pocket-ViT Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, val_accs, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Pocket-ViT Validation Accuracy')
    plt.grid(True)
    plt.savefig('validation_accuracy.png')
    plt.close()

    print('Saved plots: training_loss.png, validation_accuracy.png')

if __name__ == '__main__':
    main()
