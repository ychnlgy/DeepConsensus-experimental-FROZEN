import torch

class AssertShape(torch.nn.Module):
    def __init__(self, *size):
        super(AssertShape, self).__init__()
        self.size = size
    
    def forward(self, X):
        if X.shape[1:] != self.size:
            raise AssertionError("Shape %s does not match the expected value of %s." % (tuple(X.shape[1:], self.size)))
        return X
