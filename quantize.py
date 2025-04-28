# INT8 post-training quantisation
import torch
import torch.quantization as tq
from models import get_student_model
from utils import load_checkpoint, save_checkpoint
from data import get_dataloaders
import argparse

def quantize_model(m, calib_loader):
    m.eval().cpu()
    backend = 'fbgemm'
    m.qconfig = tq.get_default_qconfig(backend)
    prep = tq.prepare(m, inplace=False)
    # calibrate on 500 batches
    with torch.no_grad():
        for i, (x,_) in enumerate(calib_loader):
            if i>=500: break
            prep(x)
    qmod = tq.convert(prep)
    return qmod

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='student_pruned.pth')
    p.add_argument('--output',     type=str, default='student_int8.pth')
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()

    model = get_student_model(10, 'cpu')
    load_checkpoint(model, args.checkpoint, device='cpu')
    _, _, calib = get_dataloaders(args.batch_size)
    qm = quantize_model(model, calib)
    save_checkpoint(qm, args.output)
    print('quantized model saved to', args.output)

if __name__=='__main__':
    main()
