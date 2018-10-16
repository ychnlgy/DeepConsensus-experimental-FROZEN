import torch

from .ResBlock import ResBlock

class ResNet(torch.nn.Module):

    def __init__(self, kernelseq, headsize, bodysize, tailsize, layers):
        super(ResNet, self).__init__()
        
        assert layers > 0
        
        self.bodysize = bodysize
        
        if layers == 1:
            blocks = [ResBlock(kernelseq, headsize, bodysize, tailsize)]
            self.forward = self.forward_one
        
        else:
            blocks = [ResBlock(kernelseq, headsize, bodysize, bodysize)]
            blocks.extend([
                ResBlock(kernelseq, bodysize, bodysize, bodysize)
                for _ in range(layers -2)
            ])
            blocks.append(ResBlock(kernelseq, bodysize, bodysize, tailsize))
        
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, X):
        for output in self.iter_forward(X):
            continue
        return output
    
    def iter_forward(self, X):
        N, C, W, H = X.size()
        d, r = divmod(self.bodysize, C)
        assert r == 0
        W = self.blocks[0](X)
        yield W
        X = W + X.repeat(1, d, 1, 1)
        for block in self.blocks[1:-1]:
            W = block(X)
            yield W
            X = W + X
        yield self.blocks[-1](X)
    
    def forward_one(self, X):
        return self.blocks[0](X)
