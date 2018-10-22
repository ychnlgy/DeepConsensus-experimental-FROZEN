import torch

import misc

class Kernel(torch.nn.Module):

    def __init__(self, kernel, stride):
        super(Kernel, self).__init__()
        self.k = misc.param.convert2d(kernel)
        self.s = misc.param.convert2d(stride)
    
    def obtain_kernel_slices(self, U, *size):
        return U[self.kx, self.ky].view(self.ty, self.tx, self.kn, *size).transpose(0, 1)
    
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
        
        print(self.kx)
        print(self.ky)
        input()
        
        self.compute_kernel_indices = misc.util.do_nothing
