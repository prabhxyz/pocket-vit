# model setup — straight from timm
import timm
import torch.nn as nn

def get_teacher_model(num_classes=10, device='cuda'):
    # DeiT-Small pretrained
    model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=num_classes)
    model.to(device).eval()
    return model

def get_student_model(num_classes=10, device='cuda'):
    # ViT-Tiny from scratch
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    model.to(device)
    return model