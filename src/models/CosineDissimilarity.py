import torch

class CosineDissimilarity(torch.nn.CosineSimilarity):

    '''
    
    Computes element-wise cosine similarity.
    
    '''
    
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

        cosines = super(CosineSimilarity, self).forward(vectors, targets)
        return 1 - cosines.view(N, C)
    
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
