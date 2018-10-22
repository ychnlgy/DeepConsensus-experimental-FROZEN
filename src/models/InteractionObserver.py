import torch

from .Kernel import Kernel

class InteractionObserver(Kernel):

    def __init__(self):
        super(InteractionObserver, self).__init__(kernel=3, stride=1)
        self.pad = torch.nn.ZeroPad2d(1)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        
    def forward(self, X):
        X = self.pad(X)
        N, C, W, H = X.size()
        self.compute_kernel_indices(W, H)
        U = X.permute(2, 3, 0, 1)
        cos = self.cos(U, U) # W, H, N
        slices = self.obtain_kernel_slices(cos) # W, H, 9, N
        return slices.permute(3, 2, 0, 1)
        
