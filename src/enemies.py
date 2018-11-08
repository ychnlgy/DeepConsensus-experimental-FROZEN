import torch

import s2cnn

class S2ConvNet(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(S2ConvNet, self).__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 30
        b_l1 = 10
        b_l2 = 6

        grid_s2 = s2cnn.s2_near_identity_grid()
        grid_so3 = s2cnn.so3_near_identity_grid()

        self.conv1 = s2cnn.S2Convolution(
            nfeature_in=1,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        self.conv2 = s2cnn.SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3)

        self.out_layer = torch.nn.Linear(f2, f_output)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = s2cnn.so3_integrate(x)
        x = self.out_layer(x)
        return x
