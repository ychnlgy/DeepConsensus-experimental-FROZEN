import torch

import misc, models

class GlobalSumPool(torch.nn.Module, models.NormalInit):

    '''
    
    Parameters:
        g - a learnable function that transforms vectors into [0, 1]
        h - a learnable function that transforms vectors into latent space
    
    Description:
        Sums g(v_i)*h(v_i) for an entire convolution layer.
    
    '''

    def __init__(self, h, c):
        super(GlobalSumPool, self).__init__()
        self.h = h
        self.c = c
        self.init_weights(self.h)
    
    def get_init_targets(self):
        return [torch.nn.Linear]
    
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
        v = self.h(U)
        return self.c(v.sum(dim=1))
