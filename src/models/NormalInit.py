import torch

class NormalInit:

    def init_weights(self, module, miu=0, std=0.02):
        targets = self.get_init_targets()
        if type(module) in targets:
            module.weight.data.normal_(mean=miu, std=std)
        elif isinstance(module, torch.nn.Sequential):
            for submod in module:
                self.init_weights(submod, miu, std)

    def get_init_targets(self):
        raise NotImplementedError
