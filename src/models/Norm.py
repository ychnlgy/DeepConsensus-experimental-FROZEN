import torch

from .ElementPermutation import ElementPermutation

class Norm(ElementPermutation):

    '''
    
    Computes element-wise norm.
    
    '''
    
    def __init__(self, p=2):
        super(Norm, self).__init__()
        self.p = p
    
    def reduce(self, vectors, targets):
        return (vectors - targets).norm(p=self.p, dim=1)
