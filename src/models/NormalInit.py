import torch

class NormalInit:
    
    def init_weights(self, sequential):
        t = type(sequential)
        target_types = self.get_init_targets()
        if t in target_types:
            torch.nn.init.xavier_uniform(sequential.weight)
        else:
            assert isinstance(sequential, torch.nn.Sequential)
            for module in sequential:
                if type(module) in target_types:
                    torch.nn.init.xavier_uniform(module.weight)
    
    def get_init_targets(self):
        raise NotImplementedError
