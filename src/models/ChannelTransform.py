import torch

import misc
from .DenseNet import DenseNet

PERMUTATION = (2, 3, 0, 1)

class ChannelTransform(DenseNet):
    
    def forward(self, X):
        return misc.matrix.apply_permutation(self.get_net(), X, PERMUTATION)
