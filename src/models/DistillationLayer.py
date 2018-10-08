import torch, math

import misc

class DistillationLayer(torch.nn.Module):

    def __init__(self, interpreter, summarizer, channels, kernel, stride, padding):
        super(DistillationLayer, self).__init__()
        self.interpreter = interpreter
        self.summarizer = summarizer
        self.pool = torch.nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=padding, groups=channels)
        
    def forward(self, X):
        permutation = (2, 3, 0, 1)
        #X = self.interpreter(X.permute(permutation)).permute(permutation)
        X = self.pool(X)
        #X = self.summarizer(X.permute(permutation)).permute(permutation)
        return X.contiguous()
