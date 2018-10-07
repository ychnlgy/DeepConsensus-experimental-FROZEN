import torch

class InfoConv(torch.nn.Module):
    def __init__(self, channels, convs):
        super(InfoConv, self).__init__()
        
        convs, total_channels = self.create_convs(channels, convs)
        self.convs = torch.nn.ModuleList(convs)
    
    def forward(self, X):
        results = [c(X) for c in self.convs]
        return torch.cat(results, dim=1) # dim 1 refers to channels in N, C, W, H
    
    # === PRIVATE ===
    
    def create_convs(self, inchannels, convs):
        out = []
        total_channels = 0
        for outchannels, kernel in convs:
            total_channels += outchannels
            out.append(torch.nn.Conv2d(
                inchannels,
                outchannels,
                kernel,
                padding=self.calc_pad(kernel),
                stride=1
            ))
        return out, total_channels
    
    def calc_pad(self, kernel):
        if type(kernel) == int:
            return (kernel//2, kernel//2)
        else:
            return (kernel[0]//2, kernel[1]//2)
