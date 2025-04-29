import os
import argparse
from tqdm import tqdm

import torch
import torch.quantization as tq

from data import get_dataloaders
from models import get_student_model
from utils import load_checkpoint, save_checkpoint

# Force QNNPACK for quantized conv support
torch.backends.quantized.engine = 'qnnpack'

def quantize_model(model, calib_loader, device, num_calib, use_cuda):
    model.eval()
    # move model for calibration
    if use_cuda:
        model.to(device)
    else:
        model.cpu()

    # Use QNNPACK qconfig (per-channel affine supported)
    backend = 'qnnpack'
    model.qconfig = tq.get_default_qconfig(backend)
    prep = tq.prepare(model, inplace=False)

    total = min(num_calib, len(calib_loader))
    for i, (x, _) in enumerate(
        tqdm(calib_loader, total=total, desc="Calibrating", leave=False),
        start=1
    ):
        if i > num_calib:
            break
        inp = x.to(device) if use_cuda else x
        prep(inp)

    # convert to INT8
    qmodel = tq.convert(prep)
    return qmodel

def main():
    parser = argparse.ArgumentParser(
        description="INT8 PTQ with QNNPACK & progress"
    )
    parser.add_argument('--checkpoint', type=str,
                        default='student_pruned.pth',
                        help="Path to FP32 pruned model")
    parser.add_argument('--output',     type=str,
                        default='student_int8.pth',
                        help="Where to write INT8 model")
    parser.add_argument('--batch_size', type=int,
                        default=128, help="Calibration batch size")
    parser.add_argument('--num_calib',  type=int,
                        default=100, help="Calibration batches")
    parser.add_argument('--calib_cuda', action='store_true',
                        help="Run calibration on GPU")
    parser.add_argument('--device',     type=str,
                        default='cpu', choices=['cpu','cuda'],
                        help="Device for final INT8 model")
    args = parser.parse_args()

    # 1) load FP32 student
    student = get_student_model(num_classes=10, device='cpu')
    load_checkpoint(student, args.checkpoint, device='cpu')

    # 2) get calibration data
    _, _, calib_loader = get_dataloaders(args.batch_size)

    # 3) quantize
    qstudent = quantize_model(
        student,
        calib_loader,
        device=args.device,
        num_calib=args.num_calib,
        use_cuda=args.calib_cuda
    )

    # 4) save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_checkpoint(qstudent, args.output)
    print(f"✅ Quantized model saved to {args.output}")

if __name__ == '__main__':
    main()