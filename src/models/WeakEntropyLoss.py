import torch

class WeakEntropyLoss(torch.nn.Module):

    def forward(self, yh, y):
        i = torch.arange(len(yh)).to(yh.device).long()
        w = torch.ones(yh.size()).to(yh.device).float()
        w[i,y] = -1
        return (yh * w).sum(dim=1).mean()
