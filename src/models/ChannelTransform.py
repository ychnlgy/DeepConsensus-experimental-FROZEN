import torch

import misc

PERMUTATION = (2, 3, 0, 1)

class ChannelTransform(torch.nn.Module):

    def __init__(self, net):
        super(ChannelTransform, self).__init__()
        self.net = net
    
    def forward(self, X):
        return misc.matrix.apply_permutation(self.net, X, PERMUTATION)
