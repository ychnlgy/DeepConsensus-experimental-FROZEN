import torch

class DistillPool(torch.nn.Module):

    '''
    
    Parameters:
        g - a learnable function that transforms vectors into [0, 1]
        h - a learnable function that transforms vectors into latent space
    
    Description:
        Sums g(v_i)*h(v_i) for an entire convolution layer.
    
    ''''

    def __init__(self, g, h):
        super(DistillPool, self).__init__()
        self.g = g
        self.h = h
    
    def forward(self, X):
    
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
        
        Returns:
            Tensor of shape (N, C'), the latent vectors representing
            the features of the entire layer.
        
        '''
    
        N, C, W, H = X.size()
        X = X.permute(0, 2, 3, 1)
        w8s = self.g(X)
        lat = self.h(X) * w8s
        add = lat.view(N, W*H, -1).sum(dim=1)
        return add
