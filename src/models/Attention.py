import torch

class Attention(torch.nn.Module):
    
    def __init__(self, lin, parameters=None, use_input=True):
        super(Attention, self).__init__()
        self.lin = lin
        self.max = torch.nn.Softmax(dim=-1) # N, C
        
        if parameters is not None: # feature independent
            self.mem = torch.nn.Parameters(parameters)
            
            if use_input:
                self.merge = self.merge_parameters
            
            else:
                self.merge = self.get_parameters
    
    def forward(self, X):
        X = self.merge(X)
        return self.max(self.lin(X))
    
    def merge(
