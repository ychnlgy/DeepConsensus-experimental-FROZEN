import torch

from .Kernel import Kernel

class UniqueSquash(Kernel):
    def __init__(self, kernel=3, padding=1, stride=1, norm=2):
        super(UniqueSquash, self).__init__(kernel, stride)
        self.p = torch.nn.ZeroPad2d(padding)
        self.n = norm
        self.x = torch.nn.Tanh()
        self.midpt =  self.calc_midpt()
     
    def forward(self, X):
        Xp = self.p(X)
        N, C, W, H = Xp.size()
        self.compute_kernel_indices(W, H)
        U = Xp.permute(2, 3, 0, 1)
        V = self.obtain_kernel_slices(U, N, C)
        Wp, Hp, K, Np, Cp = V.size()
        Z = V[:,:,self.midpt]
        mid = Z.unsqueeze(2)
        dif = (V - mid).norm(dim=2, p=self.n)
        assert dif.size() == Z.size() == (Wp, Hp, Np, Cp)
        act = self.x(dif)
        return (Z * act).permute(2, 3, 0, 1)
    
    @staticmethod
    def unittest():
        
        squash = UniqueSquash()
        
        A = torch.Tensor([ # shape (2, 2, 2, 2)
            [
                [
                    [4, 4],
                    [4, 4]
                ],
                [
                    [4, 4],
                    [4, 4]
                ],
            ],
            [
                [
                    [2, 4],
                    [4, 4]
                ],
                [
                    [2, 4],
                    [4, 4]
                ]
            ]
        ])
        
        Y = squash(A)
        assert (Y - torch.Tensor([
            [
                [[0.0000]],
                [[0.0000]]
            ],
            [
                [[1.9961]],
                [[1.9961]]
            ]
        ])).norm().item() < 1e-3
