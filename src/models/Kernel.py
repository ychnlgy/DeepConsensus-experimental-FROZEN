import torch

import misc

class Kernel(torch.nn.Module):

    def __init__(self, kernel, stride):
        super(Kernel, self).__init__()
        self.k = misc.param.convert2d(kernel)
        self.s = misc.param.convert2d(stride)
    
    def obtain_kernel_slices(self, U, *size):
        return U[self.kx, self.ky].view(self.tx, self.ty, self.kn, *size)
    
    def compute_kernel_indices(self, W, H):
        kx, ky = self.k
        self.kn = kx * ky
        ax, ay = misc.matrix.pair_range(kx, ky)
        sx, sy = self.s
        self.tx = (W - kx) // sx + 1
        self.ty = (H - ky) // sy + 1
        nx, ny = misc.matrix.pair_range(self.tx, self.ty, sx, sy)
        self.kx = sum(misc.matrix.true_permute(nx, ax))
        self.ky = sum(misc.matrix.true_permute(ny, ay))
        
        self.compute_kernel_indices = misc.util.do_nothing
    
    @staticmethod
    def unittest():
        
        kernel = Kernel(1, 1)
        
        A = torch.arange(6).view(1, 1, 2, 3)
        kernel.compute_kernel_indices(2, 3)
        U = A.permute(2, 3, 0, 1)
        B = kernel.obtain_kernel_slices(U, 1, 1)

