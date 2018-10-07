import torch, math

import misc

class DistillationLayer(torch.nn.Module):

    def __init__(self, interpreter, summarizer, kernel, stride, padding):
        super(DistillationLayer, self).__init__()
        self.interpreter = interpreter
        self.summarizer = summarizer
        
        if padding > 0:
            self.pad = torch.nn.ReflectionPad2d(padding=padding)
        
        self.pool = torch.nn.AvgPool2d(kernel_size=kernel, stride=stride)
    
    def pad(self, X):
        return X
        
    def forward(self, X):
        out = self.pad(X)
        permutation = (2, 3, 0, 1)
        out = self.interpreter(out.permute(permutation)).permute(permutation)
        out = self.pool(out)
        out = self.summarizer(out.permute(permutation)).permute(permutation)
        return out.contiguous() # N, C'', W', H'
    
    @staticmethod
    def unittest():
    
        torch.manual_seed(5)
    
        EPS = 1e-7
    
        with torch.no_grad():
            
            # Now test the network
        
            import models
            
            X = torch.rand(3, 2, 4, 5)
            
            interpreter = models.DenseNet(headsize=2, bodysize=10, tailsize=6, layers=3, dropout=0.0, bias=False)
            interpreter.eval()
            
            summarizer = models.DenseNet(headsize=6, bodysize=10, tailsize=3, layers=3, dropout=0.0, bias=False)
            summarizer.eval()
            
            conv = DistillationLayer(
                interpreter,
                summarizer,
                5,
                stride = 2,
                padding = 2
            )
            conv.eval()
            
            result = conv(X)
            assert result.size() == (3, 3, 2, 3)
            
            # manually pad, permute, apply net and reshape
            padded = conv.pad(X)
            permuted = padded.permute(2, 3, 0, 1)
            interpreted = interpreter(permuted)
            
            # Corner (0, 0)
            slice_0_0 = interpreted[:5,:5].contiguous().view(25, 3, 6).permute(1, 0, 2).mean(dim=1)
            result_0_0 = summarizer(slice_0_0)
            assert (result_0_0.squeeze() - result[:,:,0,0].squeeze()).norm() < EPS
            
            # Corner (1, 0), note stride 2
            slice_1_0 = interpreted[2:7,:5].contiguous().view(25, 3, 6).permute(1, 0, 2).mean(dim=1)
            result_1_0 = summarizer(slice_1_0)
            assert (result_1_0.squeeze() - result[:,:,1,0].squeeze()).norm() < EPS
            
            # Corner (1, 2)
            slice_1_2 = interpreted[2:7,4:9]
            slice_1_2 = slice_1_2.contiguous().view(25, 3, 6).permute(1, 0, 2).mean(dim=1)
            result_1_2 = summarizer(slice_1_2)
            assert (result_1_2.squeeze() - result[:,:,1,2].squeeze()).norm() < EPS
