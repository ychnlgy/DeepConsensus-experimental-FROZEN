import torch

from .Kernel import Kernel

class Grouper(Kernel):
    
    def __init__(self, kernel=2, padding=0, stride=2, norm=2):
        super(Grouper, self).__init__(kernel, stride)
        self.p = torch.nn.ZeroPad2d(padding)
        self.n = norm
        self.x = torch.nn.Softmax(dim=2)
    
    def forward(self, X):
        
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
        
        Returns:
            Tensor of shape (N, C, W', H')
        
        '''
        X = self.p(X)
        N, C, W, H = X.size()
        self.compute_kernel_indices(W, H)
        U = X.permute(2, 3, 0, 1) # W, H, N, C
        norms = U.norm(p=self.n, dim=-1) # W, H, N
        
        knorms = self.obtain_kernel_slices(norms, N, 1)
        softmx = self.x(knorms)
        vector = self.obtain_kernel_slices(U, N, C)
        return (softmx * vector).sum(dim=2).permute(2, 3, 0, 1)
    
    @staticmethod
    def unittest():
        
        torch.manual_seed(5)
        
        grouper = Grouper()
        
        X1 = torch.rand(3, 2, 4, 5)
        X2 = torch.rand(3, 2, 4, 5)
        
        y1 = grouper(X1)
        y2 = grouper(X2)
        
        #raise NotImplementedError
