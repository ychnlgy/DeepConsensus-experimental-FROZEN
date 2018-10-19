import torch

class ReverseDistill(torch.nn.Module):

    def __init__(self, f, upconvs):
        super(ReverseDistill, self).__init__()
        self.f = f
        self.c = upconvs
    
    def forward(self, X):
        
        '''
        
        Given:
            X - (N, C'), distilled vectors
        
        Returns:
            (N, C, W, H), attempted reconstruction of original images.
        
        '''
