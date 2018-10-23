import torch

class DistillPool(torch.nn.Module):

    '''
    
    Parameters:
        g - a learnable function that transforms vectors into [0, 1]
        h - a learnable function that transforms vectors into latent space
    
    Description:
        Sums g(v_i)*h(v_i) for an entire convolution layer.
    
    '''

    def __init__(self, g, h, c):
        super(DistillPool, self).__init__()
        self.g = g
        self.h = h
        self.c = c
    
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
        X = X.permute(0, 2, 3, 1).view(N, W*H, C)
        v = self.h(X) * self.g(X)
        return self.c(v.sum(dim=1))
