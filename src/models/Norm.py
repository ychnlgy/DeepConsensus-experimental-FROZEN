import torch

class Norm(torch.nn.Module):

    '''
    
    Computes element-wise norm.
    
    '''
    
    def __init__(self, p=2):
        super(Norm, self).__init__()
        self.p = p
    
    def forward(self, vectors, targets):
        
        '''
        
        Given:
            vectors - (batch, dim)
            targets - (class, dim)
        
        Returns:
            predictions of shape (batch, class)
        
        '''
        
        N, D = vectors.size()
        C, D = targets.size()
        
        vectors = vectors.view(N, 1, D).repeat(1, C, 1).view(N*C, D)
        targets = targets.repeat(N, 1)
        
        assert vectors.size() == targets.size()

        diff = (vectors - targets).norm(p=self.p, dim=1)
        return diff.view(N, C)
