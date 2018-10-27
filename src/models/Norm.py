import torch

from .ElementPermutation import ElementPermutation

class Norm(ElementPermutation):

    '''
    
    Computes element-wise norm.
    
    '''
    
    def reduce(self, vectors, targets):
        return (vectors - targets)**2
