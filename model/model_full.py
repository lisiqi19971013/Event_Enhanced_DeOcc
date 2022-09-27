import torch.nn as nn
from model.EventEncoder import EventEncoder
from model.FrameEncoder import FrameEncoder, EventFrameDecoder
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

