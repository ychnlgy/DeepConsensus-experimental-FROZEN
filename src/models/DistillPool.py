import torch

class DistillPool(torch.nn.Module):

    '''
    
    Parameters:
        g - a learnable function that transforms vectors into [0, 1]
        h - a learnable function that transforms vectors into latent space
    
    Description:
        Sums g(v_i)*h(v_i) for an entire convolution layer.
    
    '''

    def __init__(self, h, c):
        super(DistillPool, self).__init__()
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
        U = X.permute(0, 2, 3, 1).view(N, W*H, C)
        v = self.h(U).sum(dim=1)
        c = self.c(v)
        _, indx = c.max(dim=1)
        neighbors = self.c.get_class_vec(indx)
        assert neighbors.size() == v.size()
        ratios = neighbors.norm(dim=1)/v.norm(dim=1)
        return c, ratios.view(-1, 1)
