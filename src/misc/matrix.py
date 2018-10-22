import torch

def apply_permutation(module, X, permutation):
    return module(X.permute(permutation)).permute(permutation)

def pair_range(x, y, dx=1, dy=1):
    ax = torch.arange(x).view(x, 1).repeat(1, y).view(-1) * dx
    ay = torch.arange(y).view(1, y).repeat(x, 1).view(-1) * dy
    return ax.long(), ay.long()

def true_permute(v1, v2):
    w1 = v1.view(-1, 1).repeat(1, len(v2)).view(-1)
    w2 = v2.view(1, -1).repeat(len(v1), 1).view(-1)
    return w1, w2
