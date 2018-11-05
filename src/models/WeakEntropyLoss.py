import torch

class WeakEntropyLoss(torch.nn.Module):

    def forward(self, yh, y):
        i = torch.arange(len(yh)).to(yh.device)
        w = torch.ones(yh.size()).to(yh.device)
        w[i,y] = -1
        return (yh * w).sum()
