import os
import sys
import glob
import argparse
import subprocess

def run(cmd):
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser(
        description="Run distill → prune → quant → eval in one go"
    )
    # distill hyperparams
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--lr',           type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_pct',   type=float, default=0.1)
    p.add_argument('--temp',         type=float, default=4.0)
    p.add_argument('--lam_kd',       type=float, default=0.7)
    p.add_argument('--lam_f',        type=float, default=0.2)
    p.add_argument('--lam_ce',       type=float, default=0.1)
    # prune hyperparams
    p.add_argument('--head_amount',  type=float, default=0.3)
    p.add_argument('--mlp_amount',   type=float, default=0.25)
    # misc
    p.add_argument('--device',       type=str,   default='cuda')
    p.add_argument('--output_dir',   type=str,   default='outputs')
    args = p.parse_args()

    python = sys.executable
    odir   = args.output_dir
    models = os.path.join(odir, 'models')
    plots  = os.path.join(odir, 'plots')
    os.makedirs(models, exist_ok=True)
    os.makedirs(plots,  exist_ok=True)

    # 1) Distillation
    cmd1 = [
        python, 'train_distill.py',
        '--epochs',       str(args.epochs),
        '--batch_size',   str(args.batch_size),
        '--lr',           str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--warmup_pct',   str(args.warmup_pct),
        '--temp',         str(args.temp),
        '--λ_kd',         str(args.lam_kd),
        '--λ_f',          str(args.lam_f),
        '--λ_ce',         str(args.lam_ce),
        '--device',       args.device,
        '--output_dir',   odir
    ]
    run(cmd1)

    # locate best checkpoint
    best_ckpts = glob.glob(os.path.join(models, 'student_best_ep*.pth'))
    if not best_ckpts:
        print("❌ No best checkpoint found in", models); sys.exit(1)
    best_ckpt = sorted(best_ckpts)[-1]
    std_ckpt = os.path.join(models, 'student_best.pth')
    # copy to standard name
    run([
        python, '-c',
        f"import shutil; shutil.copy('{best_ckpt}', '{std_ckpt}')"
    ])

    # 2) Pruning
    pruned_ckpt = os.path.join(models, 'student_pruned.pth')
    cmd2 = [
        python, 'prune.py',
        '--checkpoint', std_ckpt,
        '--output',     pruned_ckpt,
        '--head_amount',str(args.head_amount),
        '--mlp_amount', str(args.mlp_amount),
        '--device',     args.device
    ]
    run(cmd2)

    # 3) Quantization
    int8_ckpt = os.path.join(models, 'student_int8.pth')
    cmd3 = [
        python, 'quantize.py',
        '--checkpoint', pruned_ckpt,
        '--output',     int8_ckpt,
        '--batch_size', str(args.batch_size)
    ]
    run(cmd3)

    # 4) Evaluation
    cmd4 = [
        python, 'evaluate.py',
        '--fp32',       pruned_ckpt,
        '--int8',       int8_ckpt,
        '--device',     args.device,
        '--batch_size', str(args.batch_size)
    ]
    run(cmd4)

    print(f"\n✅ Pipeline complete!  Outputs in `{odir}/`")

if __name__ == '__main__':
    main()