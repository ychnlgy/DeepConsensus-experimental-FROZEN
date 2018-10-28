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
    
    def calc_midpt(self):
        kx, ky = self.k
        d, r = divmod(kx, 2)
        x = d + r - 1
        d, r = divmod(ky, 2)
        y = d + r - 1
        return x + y * kx
    
    @staticmethod
    def unittest():
        
        kernel = Kernel(1, 1)
        
        A = torch.arange(6).view(1, 1, 2, 3)
        kernel.compute_kernel_indices(2, 3)
        U = A.permute(2, 3, 0, 1)
        B = kernel.obtain_kernel_slices(U, 1, 1)
        
        assert (A.squeeze() == B.squeeze()).all()
        
        kernel = Kernel(2, 2)
        
        A = torch.arange(16).view(1, 1, 4, 4)
        kernel.compute_kernel_indices(4, 4)
        U = A.permute(2, 3, 0, 1)
        B = kernel.obtain_kernel_slices(U, 1, 1)
        
        assert (A.squeeze() == torch.LongTensor([
            [ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]
        ])).all()
        
        assert (B.squeeze() == torch.LongTensor([
            [[ 0,  1,  4,  5],
             [ 2,  3,  6,  7]],

            [[ 8,  9, 12, 13],
             [10, 11, 14, 15]]
        ])).all()

        A = torch.arange(32).view(1, 2, 4, 4)
        kernel.compute_kernel_indices(4, 4)
        U = A.permute(2, 3, 0, 1)
        B = kernel.obtain_kernel_slices(U, 1, 2)
        
        print(A)
        print(B.squeeze())
#        assert (A.squeeze() == torch.LongTensor([
#            [ 0,  1,  2,  3],
#            [ 4,  5,  6,  7],
#            [ 8,  9, 10, 11],
#            [12, 13, 14, 15]
#        ])).all()
#        
#        assert (B.squeeze() == torch.LongTensor([
#            [[ 0,  1,  4,  5],
#             [ 2,  3,  6,  7]],

#            [[ 8,  9, 12, 13],
#             [10, 11, 14, 15]]
#        ])).all()
    
