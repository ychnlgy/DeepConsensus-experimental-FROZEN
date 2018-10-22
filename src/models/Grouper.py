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

        exp_norm_X1 = torch.exp(norm_X1)
        assert (exp_norm_X1 - torch.Tensor([
        [[2.9058, 1.3368, 2.4803, 3.1116, 3.8218],
         [1.1590, 1.7373, 2.7266, 1.3047, 2.9100],
         [1.8753, 1.8647, 1.0885, 2.2717, 2.9163],
         [2.5659, 3.9722, 1.9213, 1.5560, 3.8544]],

        [[1.8644, 1.5195, 1.4471, 2.2413, 2.4449],
         [3.5835, 2.9259, 2.5651, 3.3919, 1.5121],
         [3.5404, 1.6835, 1.7827, 2.4935, 3.4180],
         [3.2301, 3.1128, 2.6684, 1.4865, 2.9445]]
        ])).norm() < 1e-3
        
        softmax = torch.nn.Softmax(dim=-1)
        softmax_X1_q1 = softmax(norm_X1[:,:2,:2].contiguous().view(-1, 4))
        softmax_X1_q2 = softmax(norm_X1[:,:2,2:4].contiguous().view(-1, 4))
        softmax_X1_q4 = softmax(norm_X1[:,2:4,2:4].contiguous().view(-1, 4))
        assert (softmax_X1_q1 - torch.Tensor([
            [0.4070, 0.1873, 0.1623, 0.2434],
            [0.1885, 0.1536, 0.3622, 0.2957]
        ])).norm() < 1e-3
        assert (softmax_X1_q2 - torch.Tensor([
            [0.2577, 0.3233, 0.2833, 0.1356],
            [0.1500, 0.2324, 0.2659, 0.3517]
        ])).norm() < 1e-3
        assert (softmax_X1_q4 - torch.Tensor([
            [0.1592, 0.3322, 0.2810, 0.2276],
            [0.2114, 0.2957, 0.3165, 0.1763]
        ])).norm() < 1e-3
        
        X1_q1 = X1[:,:,:2,:2].contiguous().view(2, 2, 4)
        X1_q2 = X1[:,:,:2,2:4].contiguous().view(2, 2, 4)
        X1_q4 = X1[:,:,2:4,2:4].contiguous().view(2, 2, 4)
        
        X1h_q1 = (softmax_X1_q1.view(2, 1, 4) * X1_q1).sum(dim=-1)
        X1h_q2 = (softmax_X1_q2.view(2, 1, 4) * X1_q2).sum(dim=-1)
        X1h_q4 = (softmax_X1_q4.view(2, 1, 4) * X1_q4).sum(dim=-1)
        
        assert (y1[:,:,0,0] - X1h_q1).norm() < 1e-3
        assert (y1[:,:,0,1] - X1h_q2).norm() < 1e-3
        assert (y1[:,:,1,1] - X1h_q4).norm() < 1e-3

