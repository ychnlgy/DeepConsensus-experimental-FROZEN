import torch

class RandomLayer(torch.nn.Module):

    def __init__(self, *size):
        super(RandomLayer, self).__init__()
        
        if size[0] == len:
            self.output = self.output_batch
            self.size = size[1:]
        else:
            self.size = size
        
        # dummy parameter so it can pretend 
        # to be a network and be optimized.
        self.p = torch.nn.Parameter(torch.ones(1))
    
    def forward(self, X):
        return self.output(X).to(X.device)
    
    # === PRIVATE ===
    
    def output(self, X):
        return torch.rand(self.size, requires_grad=True)
    
    def output_batch(self, X):
        return torch.rand(len(X), *self.size, requires_grad=True)
