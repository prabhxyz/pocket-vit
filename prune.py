# simple structured pruning script
import torch
import torch.nn.utils.prune as prune
from models import get_student_model
from utils import load_checkpoint, save_checkpoint
import argparse

def prune_model(m, head_amount=0.3, mlp_amount=0.25):
    # head pruning on qkv mats
    for n,mod in m.named_modules():
        if hasattr(mod, 'qkv'):
            prune.ln_structured(mod.qkv, 'weight', amount=head_amount, n=2, dim=1)
    # MLP fc1 prune
    for b in m.blocks:
        prune.ln_structured(b.mlp.fc1, 'weight', amount=mlp_amount, n=2, dim=0)
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='student_best.pth')
    p.add_argument('--output',     type=str, default='student_pruned.pth')
    p.add_argument('--head_amount',type=float, default=0.3)
    p.add_argument('--mlp_amount', type=float, default=0.25)
    p.add_argument('--device',     type=str,   default='cuda')
    args = p.parse_args()

    model = get_student_model(10, args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    pruned = prune_model(model, args.head_amount, args.mlp_amount)
    # remove reparam so weights stay pruned
    for m in pruned.modules():
        if isinstance(m, torch.nn.Linear):
            try: prune.remove(m, 'weight')
            except: pass
    save_checkpoint(pruned, args.output)
    print('pruned model saved to', args.output)

if __name__=='__main__':
    main()