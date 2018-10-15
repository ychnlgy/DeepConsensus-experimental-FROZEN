import torch

class AbsTanh(torch.nn.Module):

    def forward(self, X):
        return torch.tanh(X).abs()
