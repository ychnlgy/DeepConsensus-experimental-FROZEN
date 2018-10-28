import torch

import misc

EPS = 1e-8

class GravityField(torch.nn.Module):

    def __init__(self, insize, outsize):
        super(GravityField, self).__init__()
        wi, hi = misc.param.convert2d(insize)
        self.wo, self.ho = misc.param.convert2d(outsize)
        self.register_buffer("field", self.calc_field_vectors(wi, hi))
        convert = 2.0/torch.Tensor([self.wo-1, self.ho-1]).view(1, 2, 1, 1)
        self.register_buffer("convert", convert)
        self.max = torch.nn.Softmax(dim=-1)
    
    def forward(self, X):
        
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
        
        Returns:
            Tensor of shape (N, C, W', H'), after applying gravity.
        
        Description:
            Applies gravity to vectors of length C in the direction of
            the middle of the image (i.e. origin), with force proportional
            to the norm of the vector and its distance from origin.
        
        '''
        N, C, W, H = X.size()
        w = torch.tanh(X.norm(dim=1))
        #w = (w/w.view(N, W*H).sum(dim=-1).view(N, 1, 1)).view(N, 1, W, H)
        d = self.field * (1-w)
        z = self.convert_out(d)
        return self.insert_vecs(X, z)
    
    def insert_vecs(self, X, z):
        
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
            z - Tensor of shape (N, 2, W, H), discrete coordinates for output size.
        
        Returns:
            Tensor of shape (N, C, W', H'), gravitated vectors, combined with softmax on norms.
        
        '''
        N, C, W, H = X.size()
        out = torch.zeros(self.wo, self.ho, N, W, H, C).to(X.device)
        X = X.permute(2, 3, 0, 1)
        z = z.permute(2, 3, 1, 0)
        n = torch.arange(N).long().to(X.device)
        for i in range(W):
            for j in range(H):
                x, y = z[i, j, 0], z[i, j, 1]
                out[x, y, n, i, j] = X[i,j,n]
        
        out = out.view(self.wo, self.ho, N, W*H, C)
        return self.combine(out)
    
    def combine(self, out):
        
        '''
        
        Given:
            out - Tensor of shape (W', H', N, W*H, C)
        
        Returns:
            Tensor of shape (N, C, W', H'), softmax on channel vector norm.
        
        '''
        
        norms = out.norm(dim=-1) # W', H', N, W*H
        weights = self.max(norms).unsqueeze(-1)
        vecs = (weights * out).sum(dim=3) # W', H', N, C
        return vecs.permute(2, 3, 0, 1)
    
    def convert_out(self, d):
        
        '''
        
        Given:
            D - Tensor of shape (N, 2, W, H)
        
        Returns:
            Tensor of shape (N, 2, W, H), discrete coordinates for output size.
        
        '''
        return ((d+1.0)/self.convert).round().long()

    def calc_field_vectors(self, w, h):
        mw, mh = w//2, h//2
        x, y = misc.matrix.pair_range(w, h)
        x = x.float() / mw - 1 # range from -1 to 1
        y = y.float() / mh - 1
        v = torch.stack([x, y])
        n = v.norm(dim=0).view(1, -1)
        n[n < EPS] = 1.0
        v = v/n
        return v.view(1, 2, w, h)
    
    @staticmethod
    def unittest():
        field = GravityField(5, 3)
        
        A = torch.Tensor([ # 2, 2, 5, 5
            [
                [
                    [0, 0, 0, 0, 100],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 100],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ],
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 50],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 50],
                ]
            ]
        ])
        
        P = field(A)
        assert (P - torch.Tensor([
        [[[  0.,   0.,   0.],
          [  0., 100.,   0.],
          [  0.,   0.,   0.]],

         [[  0.,   0.,   0.],
          [  0., 100.,   0.],
          [  0.,   0.,   0.]]],


        [[[  0.,   0.,   0.],
          [  0.,  50.,   0.],
          [  0.,   0.,   0.]],

         [[  0.,   0.,   0.],
          [  0.,  50.,   0.],
          [  0.,   0.,   0.]]]
        ])).norm() < 1e-3

