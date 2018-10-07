import torch

import misc

class Reshape(torch.nn.Module):
    def __init__(self, *size, contiguous=False):
        super(Reshape, self).__init__()
        
        if not contiguous:
            self.make_contiguous = self._make_contiguous
        
        if size[0] == len:
            self.view = self.view_batch
            self.size = size[1:]
        
        else:
            self.size = size
    
    def forward(self, X):
        return self.view(self.make_contiguous(X))
    
    # === PRIVATE ===
    
    def make_contiguous(self, X):
        return X.contiguous()
    
    def _make_contiguous(self, X):
        return X
    
    def view(self, X):
        return X.view(self.size)
    
    def view_batch(self, X):
        return X.view(len(X), *self.size)
