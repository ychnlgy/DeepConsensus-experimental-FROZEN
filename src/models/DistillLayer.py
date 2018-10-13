import torch

class DistillLayer(torch.nn.Module):

    def __init__(self, convlayer, counter, summarizer):
        super(DistillLayer, self).__init__()
        self.convlayer  = convlayer
        self.counter    = counter
        self.summarizer = summarizer
    
    def forward(self, X):
        N, C, W, H = X.size()
        convout = self.convlayer(X)
        infovec = convout.permute(0, 2, 3, 1).view(N, W*H, C)
        counted = self.counter(infovec).mean(dim=1)
        summary = self.summarizer(counted)
        return convout, summary
