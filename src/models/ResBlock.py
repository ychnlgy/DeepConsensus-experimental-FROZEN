import torch

class ResBlock(torch.nn.Module):
    
    def __init__(self, kernelseq, headsize, bodysize, tailsize):
        super(ResBlock, self).__init__()
        
        if len(kernelseq) == 1:
            net = [self.create_unit(headsize, tailsize, kernelseq[0])]
        else:
            net = [self.create_unit(headsize, bodysize, kernelseq[0])]
            net.extend([
                self.create_unit(bodysize, bodysize, k)
                for k in kernelseq[1:-1]
            ])
            net.append(self.create_unit(bodysize, tailsize, kernelseq[-1]))
        
        self.net = torch.nn.Sequential(*net)
        print(self.net)
    
    def forward(self, X):
        return self.net(X)
    
    def create_unit(self, c_in, c_out, kernel):
        groups = self.calc_groups(c_in, c_out)
        return torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, kernel, padding=kernel//2, groups=groups),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(c_out)
        )
    
    def calc_groups(self, c1, c2):
        c = min(c1, c2)
        # if c1 and c2 is divisble by c, their remainders is 0.
        return [c, 1][bool((c1 % c) & (c2 % c))]
