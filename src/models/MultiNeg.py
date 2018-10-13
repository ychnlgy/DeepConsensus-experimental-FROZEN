import torch

class MultiNeg(torch.nn.Module):
    
    def __init__(self, voters):
        super(MultiNeg, self).__init__()
        self.voters = torch.nn.ModuleList(voters)
        self.min = torch.nn.Softmin(dim=0)
    
    def forward(self, X):
        out = [v(X) for v in self.voters]
        out = torch.cat(out, dim=1) # N, predictions (number of columns should be classes)
        return self.min(out)
