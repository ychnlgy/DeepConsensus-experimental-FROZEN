import torch

class ReluTanh(torch.nn.Module):
    
    def forward(self, X):
        return torch.nn.functional.relu(torch.tanh(X))
