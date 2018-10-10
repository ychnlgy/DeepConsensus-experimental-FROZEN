import torch

class DistillationBlock(torch.nn.Module):

    def __init__(self, cnn, atn, lin, weight):
        super(DistillationBlock, self).__init__()
        self.cnn = cnn
        self.atn = atn
        self.lin = lin
        self.lam = weight
        
    def forward(self, X):
    
        '''
        
        Returns the 
            [1] convolution and 
            [2] prediction using the attended, summed infovectors.
        
        '''
    
        out = self.cnn(X)
        atn, out = self.atn(out)
        lin = self.lin(atn) * self.lam
        return out, lin
