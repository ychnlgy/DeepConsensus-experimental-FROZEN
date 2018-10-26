import torch

from .ElementPermutation import ElementPermutation

class CosineSimilarity(ElementPermutation):

    '''
    
    Computes element-wise cosine similarity.
    
    '''
    
    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.cos = torch.nn.CosineSimilarity(dim=1)
    
    def reduce(self, vectors, targets):
        return self.cos(vectors, targets)
    
    @staticmethod
    def unittest():
        
        cosine = CosineSimilarity()
        
        vectors = torch.Tensor([
            [1, 1, 1],
            [2, 2, 2],
            [1, 0, 0],
            [2, 0, 0],
            [3, 4, 5],
            [9, 16, 25]
        ])
        
        targets = torch.Tensor([
            [5, 0, 0],
            [81, 256, 625],
            [-5, -5, -5]
        ])
        
        assert (cosine(vectors, targets) - torch.Tensor([
            [ 0.5774,  0.8165, -1.0000],
            [ 0.5774,  0.8165, -1.0000],
            [ 1.0000,  0.1191, -0.5774],
            [ 1.0000,  0.1191, -0.5774],
            [ 0.4243,  0.9131, -0.9798],
            [ 0.2902,  0.9693, -0.9307]
        ])).norm().item() < 1e-2
