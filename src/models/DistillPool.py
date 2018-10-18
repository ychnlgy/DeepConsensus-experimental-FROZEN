import torch

class DistillPool(torch.nn.Module):

    '''
    
    Parameters:
        g - a learnable function that transforms vectors into [0, 1]
        h - a learnable function that transforms vectors into latent space
    
    Description:
        Sums g(v_i)*h(v_i) for an entire convolution layer.
    
    '''

    def __init__(self, g, h):
        super(DistillPool, self).__init__()
        self.g = g
        self.h = h
    
    def forward(self, X, summary=None):
    
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
        w = self.g(self.combine(X, summary))
        v = self.h(X)
        return (w * v).sum(dim=1)
    
    def combine(self, X, summary):
        if summary is None:
            return X
        else:
            B, N, C = X.size()
            M, K = summary.size()
            assert B == M
            s = summary.view(M, 1, K).repeat(1, N, 1)
            o = torch.cat([X, s], dim=-1)
            assert o.size() == (B, N, C + K)
            return o
