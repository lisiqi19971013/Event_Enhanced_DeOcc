'''
SNN Encoder
Event Encoder
下采样的Feature过conv之后再concat到上采样
CondConv
Decoder时候加先验，用event surface concat进来
'''
import torch.nn as nn
from model.EventEncoder import EventEncoder, EventEncoder1
from model.FrameEncoder import FrameEncoder, EventFrameDecoder
import torch.nn.functional as F
import torch


class model(nn.Module):
    def __init__(self, netParams, inChannels, outChannels=3, layers=[128, 128, 256, 256, 512, 512, 512], norm='BN'):
        super().__init__()
        self.EventEncoder = EventEncoder(netParams, layers=layers, norm=norm)
        self.FrameEncoder = FrameEncoder(inChannels, layers=layers, norm=norm)
        self.EventFrameDecoder = EventFrameDecoder(outChannels, layers=layers, norm=norm)

    def forward(self, event, frames):
        f_e = self.EventEncoder(event)
        f_f = self.FrameEncoder(frames)
        output = self.EventFrameDecoder(f_e, f_f)
        return output


class model1(nn.Module):
    def __init__(self, netParams, inChannels, outChannels=3, layers=[128, 128, 256, 256, 512, 512, 512], norm='BN'):
        super().__init__()
        self.EventEncoder = EventEncoder1(netParams, layers=layers, norm=norm)
        self.FrameEncoder = FrameEncoder(inChannels, layers=layers, norm=norm)
        self.EventFrameDecoder = EventFrameDecoder(outChannels, layers=layers, norm=norm)

    def forward(self, event, frames):
        f_e = self.EventEncoder(event)
        f_f = self.FrameEncoder(frames)
        output = self.EventFrameDecoder(f_e, f_f)
        return output


if __name__ == '__main__':
    nb_of_time_bin = 15
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
    m = model(netParams, 33, 3).cuda()
    e = torch.zeros([2, 2, 256, 256, 30]).cuda()
    f = torch.zeros([2, 33, 256, 256]).cuda()
    o = m(e, f)