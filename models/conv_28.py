from torch import nn

from utilities import nn_utilities as n_util


class Model(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, **kwargs):
        super(Model, self).__init__()
        self.ngpu = kwargs.get('ngpu', 1)

        activation_setting = kwargs.get('activation')
        activation = []
        if activation_setting:
            activation_args = activation_setting.get('args', {})
            activation.append(getattr(nn, activation_setting['name'])(**activation_args))
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels * 2, mid_channels * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(mid_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels * 4, mid_channels * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(mid_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels * 8, out_channels, 4, 1, 0, bias=False),
            *activation
        )

    def forward(self, x):
        output = n_util.data_parallel_model(self.cnn, x, self.ngpu)
        return output.view(x.size(0), -1)


if __name__ == '__main__':
    import torch
    model = Model(1, 10, 20, activation='Sigmoid')
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    print(y)
    print(y.size())
