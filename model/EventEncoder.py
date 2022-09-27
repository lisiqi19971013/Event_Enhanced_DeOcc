from torch import nn
import slayerSNN as snn
import torch
from model import down_light, SizeAdapter, CondConv2D
import torch.nn.functional as F


def getNeuronConfig(type: str='SRMALPHA', theta: float=10., tauSr: float=1., tauRef: float=1., scaleRef: float=2., tauRho: float=0.3, scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }


class SnnEncoder(nn.Module):
    def __init__(self, netParams, hidden_number=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4], scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100]):
        super(SnnEncoder, self).__init__()

        self.neuron_config = []
        self.neuron_config.append(getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0], scaleRho=scaleRho[0]))
        self.neuron_config.append(getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1], scaleRho=scaleRho[1]))
        self.neuron_config.append(getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2], scaleRho=scaleRho[2]))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams)
        self.slayer2 = snn.layer(self.neuron_config[1], netParams)
        self.slayer3 = snn.layer(self.neuron_config[2], netParams)

        self.conv1 = self.slayer1.conv(2, 16, kernelSize=3, padding=1)
        self.conv2 = self.slayer2.conv(18, 16, kernelSize=3, padding=1)
        self.conv3 = self.slayer3.conv(18, hidden_number, kernelSize=1, padding=0)

    def forward(self, spikeInput):
        psp0 = self.slayer1.psp(spikeInput)
        psp1 = self.conv1(psp0)
        spikes_1 = self.slayer1.spike(psp1)

        psp2 = torch.cat([self.slayer2.psp(spikes_1), psp0], dim=1)
        psp2 = self.conv2(psp2)
        spikes_2 = self.slayer2.spike(psp2)

        psp3 = torch.cat([self.slayer3.psp(spikes_2), psp0], dim=1)
        psp3 = self.conv3(psp3)
        return psp3


class EventEncoder(nn.Module):
    def __init__(self, netParams, hidden_number=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], layers=[128, 128, 128, 256, 256, 512, 512], norm=False):
        super(EventEncoder, self).__init__()
        self.SNN = SnnEncoder(netParams, hidden_number, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        self.IN = nn.BatchNorm2d(hidden_number)

        self.layers = layers
        self.conv1 = nn.Sequential(CondConv2D(hidden_number, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)) if norm else nn.Conv2d(hidden_number, 64, 3, stride=1, padding=1)
        self.down0 = down_light(64, self.layers[0], norm=norm)
        for k in range(1, len(self.layers)):
            setattr(self, 'down%d'%k, down_light(self.layers[k-1], self.layers[k], norm=norm))

    def forward(self, events):
        bs, _, H, W, Ts = events.shape
        snn_fea = self.SNN(events)
        snn_fea = torch.mean(snn_fea, dim=-1)
        snn_fea = self.IN(snn_fea)
        output = []

        x = F.leaky_relu(self.conv1(snn_fea), negative_slope=0.1)
        output.append(x)
        for k in range(len(self.layers)):
            x = getattr(self, 'down%d'%k)(x)
            output.append(x)
        return output


if __name__ == '__main__':
    nb_of_time_bin = 15
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
    m = EventEncoder(netParams)
    x = torch.zeros([1, 2, 256, 256, 30]).cuda()
    m.cuda()
    o = m(x)