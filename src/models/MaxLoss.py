import torch

import misc

class MaxLoss(torch.nn.CrossEntropyLoss):

    def forward(self, yh, y):
        vals, indx = yh.max(dim=1)
        batch = torch.arange(len(yh)).to(yh.device)
        blank = torch.zeros_like(yh, requires_grad=True).to(yh.device)
        blank[batch, indx] = vals
        return super(MaxLoss, self).forward(blank, y)
