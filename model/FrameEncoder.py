import torch.nn as nn
from model import up1, down_light, CondConv2D
import torch.nn.functional as F
import torch


class FrameEncoder(nn.Module):
    def __init__(self, inChannels, size_adapter=None, layers=[128, 128, 256, 256, 512, 512, 512], norm='BN'):
        super().__init__()
        self._size_adapter = size_adapter
        self.conv1 = nn.Sequential(CondConv2D(inChannels, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)) if norm else CondConv2D(inChannels, 64, 3, stride=1, padding=1)
        self.layers = layers

        self.down0 = down_light(64, self.layers[0], norm=norm)
        for k in range(1, len(self.layers)):
            setattr(self, 'down%d'%k, down_light(self.layers[k-1], self.layers[k], norm=norm))

    def forward(self, x):
        output = []
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        output.append(x)

        for k in range(len(self.layers)):
            x = getattr(self, 'down%d' % k)(x)
            output.append(x)

        return output


class EventFrameDecoder(nn.Module):
    def __init__(self, outChannels, size_adapter=None, layers=[128,128,256,256,512,512,512], norm='BN'):
        super(EventFrameDecoder, self).__init__()
        self._size_adapter = size_adapter
        num_heads = [8,8,4,4,2,2,1]
        for k in range(1, len(layers)):
            setattr(self, 'up%d'%k, up1(layers[k], layers[k-1], num_heads[k], norm))

        self.up0 = up1(layers[0], 64, num_heads[0], norm)
        self.conv_out = CondConv2D(64, outChannels, kernel_size=3, stride=1, padding=1)
        self.layers = layers

        for k in range(1, len(layers)):
            if norm:
                setattr(self, 'fuse%d'%k, nn.Sequential(nn.Conv2d(layers[k-1]*2, int(layers[k-1]/2), 3, padding=1), nn.BatchNorm2d(int(layers[k-1]/2)), nn.LeakyReLU(negative_slope=0.1),
                                                        nn.Conv2d(int(layers[k-1]/2), layers[k-1], 3, padding=1), nn.BatchNorm2d(layers[k-1]), nn.LeakyReLU(negative_slope=0.1)))
            else:
                setattr(self, 'fuse%d'%k, nn.Sequential(nn.Conv2d(layers[k-1]*2, int(layers[k-1]/2), 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
                                                        nn.Conv2d(int(layers[k-1]/2), layers[k-1], 3, padding=1), nn.LeakyReLU(negative_slope=0.1)))
        self.fuse0 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.1)) if norm else \
            nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.LeakyReLU(negative_slope=0.1), nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1))

    def forward(self, input1, input2):
        x = torch.mul(input1[-1], input2[-1])

        for k in range(1, len(self.layers)+1):
            x1 = torch.cat([input1[-1-k], input2[-1-k]], dim=1)
            x2 = getattr(self, 'fuse%d'%(len(self.layers)-k))(x1) + input1[-1-k] + input2[-1-k]
            x = getattr(self, 'up%d'%(len(self.layers)-k))(x, x2)

        x = self.conv_out(x)
        return x


# if __name__ == '__main__':
#     e = FrameEncoder(33)
#     d = SingleDecoder(3)
#
#     x = torch.zeros([2,33,256,256])
#     o = e(x)
#     op = d(o)
