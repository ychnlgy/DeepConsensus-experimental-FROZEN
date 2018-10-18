import torch

import misc

class DistillNet(torch.nn.Module):

    def __init__(self, iternet, pools):
        super(DistillNet, self).__init__()
        self.iternet = iternet
        self.pools = torch.nn.ModuleList(pools)
    
    def forward(self, X):
        # list of (N, C, W, H) from topmost to bottom
        layers = misc.util.reverse_iterator(self.iternet.iter_forward(X))
        
        summary = None
        for pool, layer in zip(self.pools, layers):
            summary = pool(layer, summary) # (N, C')
        return summary
