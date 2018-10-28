import torch, math

import misc

from .Classifier import Classifier
from .ChannelTransform import ChannelTransform
from .UniqueSquash import UniqueSquash
from .SoftmaxCombine import SoftmaxCombine

class DistillPool(torch.nn.Module):

    '''
    
    Parameters:
        g - a learnable function that transforms vectors into [0, 1]
        h - a learnable function that transforms vectors into latent space
    
    Description:
        Sums g(v_i)*h(v_i) for an entire convolution layer.
    
    '''

    def __init__(self, layers, channels, classes):
        super(DistillPool, self).__init__()
        self.layers = layers
        self.transformers = self.replicate(ChannelTransform,
                headsize = channels,
                bodysize = channels,
                tailsize = channels,
                layers = 1
        )
        self.classifier = Classifier(hiddensize=channels, classes=classes)
        self.squash = self.replicate(UniqueSquash,
            kernel = 3,
            padding = 1,
            stride = 1
        )
        self.combine = self.replicate(SoftmaxCombine,
            kernel = 2,
            padding = 0,
            stride = 2
        )
        #self.batchnorms = self.replicate(torch.nn.BatchNorm2d, channels)
        self.max = torch.nn.Softmax(dim=-1)
    
    def replicate(self, Module, *args, **kwargs):
        return torch.nn.ModuleList([Module(*args, **kwargs) for i in range(self.layers)])
    
    def forward(self, X):
    
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
            summary - Tensor of shape (N, C'), pooled vectors of the upper layer
        
        Returns:
            Tensor of shape (N, C'), the latent vectors representing
            the features of the entire layer.
        
        '''
        for i in range(self.layers):
            X = self.squash[i](X)
            X = self.combine[i](X)
            X = self.transformers[i](X)
        N, C, W, H = X.size()
        X = X.view(N, C, W*H)
        X = (X * self.max(X)).sum(dim=-1)
        assert X.size() == (N, C)
        return self.classifier(X)
