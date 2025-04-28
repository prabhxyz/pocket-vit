# eval + benchmark
import torch, time, os
from data import get_dataloaders
from models import get_student_model
from utils import load_checkpoint, accuracy
import argparse

def eval_model(m, loader, device):
    m.eval().to(device)
    total=0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            total += accuracy(m(x), y)
    return total/len(loader)

def benchmark_ts(ts_path, device='cpu'):
    m = torch.jit.load(ts_path).to(device)
    dummy = torch.randn(1,3,224,224).to(device)
    for _ in range(10): m(dummy)
    t0=time.time()
    for _ in range(100): m(dummy)
    t1=time.time()
    return 100/(t1-t0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fp32', type=str, default='student_pruned.pth')
    p.add_argument('--int8', type=str, default='student_int8.pth')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=128)
    args=p.parse_args()

    _, val, _ = get_dataloaders(args.batch_size)
    m_fp = get_student_model(10, args.device)
    load_checkpoint(m_fp, args.fp32, args.device)
    m_int = get_student_model(10, 'cpu')
    load_checkpoint(m_int, args.int8, 'cpu')

    print('FP32 Acc:', eval_model(m_fp, val, args.device))
    print('INT8 Acc:', eval_model(m_int, val, 'cpu'))

    # if you jit-trace and save as student_int8_ts.pt
    if os.path.exists('student_int8_ts.pt'):
        print('INT8 TorchScript FPS:', benchmark_ts('student_int8_ts.pt','cpu'))

if __name__=='__main__':
    main()