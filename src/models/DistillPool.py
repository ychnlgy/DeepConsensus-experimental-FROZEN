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
        X = self.h(U)
        s = X.sum(dim=1)
        print(s[0])
        c = self.c(s)
        #_, indx = c.max(dim=1)
        #r = self.c.get_class_vec(indx).view(N, -1).abs()/s.abs()
        #X = X * r.view(N, 1, -1)
        return c, X.permute(0, 2, 1).view(N, -1, W, H)
