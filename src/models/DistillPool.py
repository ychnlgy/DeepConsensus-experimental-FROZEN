import torch

import misc
from .AbsTanh import AbsTanh

class DistillPool(torch.nn.Module):

    '''
    
    Parameters:
        g - a learnable function that transforms vectors into [0, 1]
        h - a learnable function that transforms vectors into latent space
    
    Description:
        Sums g(v_i)*h(v_i) for an entire convolution layer.
    
    '''

    def __init__(self, channels, h, c):
        super(DistillPool, self).__init__()
        self.h = h
        self.c = c
        self.w = torch.nn.Parameter(torch.rand(1, 1, channels))
        self.t = AbsTanh()
        self.x = torch.nn.Softmax(dim=-1)
    
    def forward(self, X):
    
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
            summary - Tensor of shape (N, C'), pooled vectors of the upper layer
        
        Returns:
            Tensor of shape (N, C'), the latent vectors representing
            the features of the entire layer.
        
        '''
    
        N, C, W, H = X.size()
        U = X.permute(0, 2, 3, 1).view(N, W*H, C)
        w = self.x(self.w)
        v = self.h(U * w)
        s = self.t(v).sum(dim=1)
        misc.debug.println(s[0])
        return self.c(s)
