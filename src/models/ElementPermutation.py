import torch

class ElementPermutation(torch.nn.Module):
    
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
        
        self.N = N
        self.C = C
        
        vectors = vectors.view(N, 1, D).repeat(1, C, 1).view(N*C, D)
        targets = targets.repeat(N, 1)
        
        assert vectors.size() == targets.size()

        reduced = self.reduce(vectors, targets)
        return reduced.view(N, C)
    
    def get_NC(self):
        return self.N, self.C
    
    def reduce(self, vectors, targets):
        raise NotImplementedError
