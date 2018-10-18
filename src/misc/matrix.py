import torch

def apply_permutation(module, X, permutation):
    return module(X.permute(permutation)).permute(permutation)

def number(vec):
    ind = torch.arange(len(vec))
    vec = torch.stack([ind, vec])
    return vec.transpose(0, 1)
