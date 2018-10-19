import torch

def paramcount(model):
    return sum(map(torch.numel, model.parameters()))
