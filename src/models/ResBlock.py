import torch

LEAKY = torch.nn.LeakyReLU()
EMPTY = torch.nn.Sequential()

class ResBlock(torch.nn.Module):
    
    def __init__(self, kernelseq, headsize, bodysize, tailsize):
        super(ResBlock, self).__init__()
        
        if len(kernelseq) == 1:
            net = [self.create_unit(headsize, tailsize, kernelseq[0], EMPTY)]
        else:
            net = [self.create_unit(headsize, bodysize, kernelseq[0])]
            net.extend([
                self.create_unit(bodysize, bodysize, k)
                for k in kernelseq[1:-1]
            ])
            net.append(self.create_unit(bodysize, tailsize, kernelseq[-1], EMPTY))
        
        self.act = LEAKY
        self.net = torch.nn.Sequential(*net)
    
    def forward(self, X):
        out = self.net(X)
        add = self.add(out, X)
        return self.act(add)
    
    def add(self, out, X):
        C0 = X.size(1)
        Cf = out.size(1)
        d, r = divmod(Cf, C0)
        
        add = X.repeat(1, d, 1, 1)
        if r > 0:
            add = torch.cat([add, X[:,:r]], dim=1)

        assert add.size(1) == Cf
        return out + add
    
    def create_unit(self, c_in, c_out, kernel, activation=LEAKY):
        groups = self.calc_groups(c_in, c_out)
        return torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, kernel, padding=kernel//2, groups=groups),
            activation,
            torch.nn.BatchNorm2d(c_out)
        )
    
    def calc_groups(self, c1, c2):
        c = min(c1, c2)
        # if c1 and c2 is divisble by c, their remainders is 0.
        return [c, 1][bool((c1 % c) + (c2 % c))]
