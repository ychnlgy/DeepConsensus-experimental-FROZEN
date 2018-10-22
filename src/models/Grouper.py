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
        
        X1 = torch.rand(2, 2, 4, 5)
        X2 = torch.rand(2, 2, 4, 5)
        
        y1 = grouper(X1)
        y2 = grouper(X2)
        
        norm_X1 = X1.norm(dim=1)
        assert (norm_X1 - torch.Tensor([
        [[1.0667, 0.2903, 0.9084, 1.1351, 1.3407],
         [0.1475, 0.5524, 1.0030, 0.2660, 1.0682],
         [0.6288, 0.6231, 0.0848, 0.8205, 1.0703],
         [0.9423, 1.3793, 0.6530, 0.4421, 1.3492]],

        [[0.6230, 0.4184, 0.3696, 0.8070, 0.8940],
         [1.2763, 1.0736, 0.9420, 1.2214, 0.4135],
         [1.2642, 0.5209, 0.5781, 0.9137, 1.2291],
         [1.1725, 1.1355, 0.9815, 0.3964, 1.0799]]
        ])).norm() < 1e-3

