import torch

class Counter(torch.nn.Module):
    
    def forward(self, X):
    
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
        
        Returns:
            Tensor of shape (N, C), where the features are
            counted and summed to a single channel vector.
        
        '''
    
        N, C, W, H = X.size()
        X = X.view(N, C, W*H)
        X = torch.tanh(X)**2
        X = X.sum(dim=-1)
        assert X.size() == (N, C)
        return X
