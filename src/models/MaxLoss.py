import torch

class MaxLoss(torch.nn.CrossEntropyLoss):

    def forward(self, yh, y):
        vals, indx = yh.max(dim=1)
        blank = torch.zeros_like(yh, requires_grad=True)
        blank[indx] = vals
        return super(MaxLoss, self).forward(blank, y)
