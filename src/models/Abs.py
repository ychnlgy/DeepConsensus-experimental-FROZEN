import torch

class Abs(torch.nn.Module):

    def forward(self, X):
        return X.abs()
