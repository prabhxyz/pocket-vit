# misc utils
import torch
import torch.nn.functional as F

def accuracy(output, target):
    # top-1
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device))
