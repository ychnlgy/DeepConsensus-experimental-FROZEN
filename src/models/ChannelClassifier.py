import torch

from .Homogenizer import Homogenizer

class ChannelClassifier(Homogenizer):
    
    def forward(self, X):
    
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
        
        Returns:
            Tensor of shape (N, classes, W, H)
        
        '''
    
        N, C, W, H = X.size()
        X = X.permute(0, 2, 3, 1).contiguous().view(N*W*H, C)
        U = super(ChannelClassifier, self).forward(X) # N*W*H, C'
        Nh, Ch = W.size()
        U = U.view(N, W, H, Ch).permute(0, 3, 1, 2)
        return U
