import torch

class DistillLayer(torch.nn.Module):

    def __init__(self, convlayer, counter, summarizer):
        super(DistillLayer, self).__init__()
        self.convlayer  = convlayer
        self.counter    = counter
        self.summarizer = summarizer
    
    def forward(self, X):
        convout = self.convlayer(X)
        N, C, W, H = convout.size()
        infovec = convout.permute(0, 2, 3, 1).view(N, W*H, C)
        N, L, C = infovec.size()
        counted = self.counter(infovec).mean(dim=1)
        assert counted.size() == (N, C)
        summary = self.summarizer(counted)
        return convout, summary
