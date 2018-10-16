import torch

import misc

class DistillLayer(torch.nn.Module):

    def __init__(self, convlayer, masker, selector, dropout, interpreter, summarizer):
        super(DistillLayer, self).__init__()
        self.convlayer   = convlayer
        self.masker      = masker
        self.selector    = selector
        self.interpreter = interpreter
        self.summarizer  = summarizer
        self.dropout     = torch.nn.Dropout2d(p=dropout)
    
    def forward(self, X):
        convout = self.convlayer(X)
        convinp = self.dropout(convout)
        convinp = convinp.permute(0, 2, 3, 1) # N, W, H, C
        N, W, H, C = convinp.size()
        convinp = convinp.view(N, W*H, C)
        maskout = self.masker(convinp)
        selects = self.selector(convinp)
        interpd = self.interpreter(convinp)
        spooled = (maskout * interpd * selects).sum(dim=1)
        summary = self.summarizer(spooled)
        return convout, summary
