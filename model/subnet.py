import datetime
import math
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import numpy as np


def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)


class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """
    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        # print(self._pixels_pad_to_height, self._pixels_pad_to_width)
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]


class TimeRecorder(object):
    def __init__(self, total_epoch, iter_per_epoch):
        self.total_epoch = total_epoch
        self.iter_per_epoch = iter_per_epoch
        self.start_train_time = datetime.datetime.now()
        self.start_epoch_time = datetime.datetime.now()
        self.t_last = datetime.datetime.now()

    def get_iter_time(self, epoch, iter):
        dt = (datetime.datetime.now() - self.t_last).__str__()
        self.t_last = datetime.datetime.now()
        remain_time = self.cal_remain_time(epoch, iter, self.total_epoch, self.iter_per_epoch)
        end_time = (datetime.datetime.now() + datetime.timedelta(seconds=remain_time)).strftime("%Y-%m-%d %H:%S:%M")
        remain_time = datetime.timedelta(seconds=remain_time).__str__()
        return dt, remain_time, end_time

    def cal_remain_time(self, epoch, iter, total_epoch, iter_per_epoch):
        t_used = (datetime.datetime.now() - self.start_train_time).total_seconds()
        time_per_iter = t_used / (epoch * iter_per_epoch + iter + 1)
        remain_iter = total_epoch * iter_per_epoch - (epoch * iter_per_epoch + iter + 1)
        remain_time_second = time_per_iter * remain_iter
        return remain_time_second


class up(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(up, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, padding=1, dilation=1), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 1, 1, 0)
        self.conv3 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, padding=3, dilation=3), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 3, 1, 1)
        self.conv5 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, padding=5, dilation=5), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 5, 1, 2)
        self.conv7 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, padding=7, dilation=7), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 7, 1, 3)
        self.conv = nn.Sequential(nn.Conv2d(outChannels, outChannels, 1, 1, 0), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels, outChannels, 3, 1, 1)
        self.conv_out = nn.Sequential(nn.Conv2d(outChannels*3, outChannels, 3, 1, 1), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels*3, outChannels, 3, 1, 1)

    def forward(self, x, skpCn1, skpCn2):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        x_1 = self.conv1(x)
        x_3 = self.conv3(x)
        x_5 = self.conv5(x)
        x_7 = self.conv7(x)
        x = F.leaky_relu(torch.cat((x_1, x_3, x_5, x_7), 1), negative_slope=0.1)
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        x = torch.cat((x, skpCn1, skpCn2), 1)

        x = F.leaky_relu(self.conv_out(x), negative_slope=0.1)

        return x


class up1(nn.Module):
    def __init__(self, inChannels, outChannels, num_heads=0,norm='BN'):
        super(up1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, padding=1, dilation=1), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 1, 1, 0)
        self.conv3 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, padding=3, dilation=3), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 3, 1, 1)
        self.conv5 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, padding=5, dilation=5), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 5, 1, 2)
        self.conv7 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, padding=7, dilation=7), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 7, 1, 3)
        self.conv = nn.Sequential(nn.Conv2d(outChannels, outChannels, 1, 1, 0), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels, outChannels, 3, 1, 1)
        self.conv_out = nn.Sequential(nn.Conv2d(outChannels*2, outChannels, 3, 1, 1), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels*3, outChannels, 3, 1, 1)
        # self.att = Attention(inChannels, num_heads, bias=True)
        # self.bn = nn.BatchNorm2d(inChannels)

    def forward(self, x, skpCn1):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        # x = self.bn(self.att(x))

        x_1 = self.conv1(x)
        x_3 = self.conv3(x)
        x_5 = self.conv5(x)
        x_7 = self.conv7(x)
        x = F.leaky_relu(torch.cat((x_1, x_3, x_5, x_7), 1), negative_slope=0.1)
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        x = torch.cat((x, skpCn1), 1)

        x = F.leaky_relu(self.conv_out(x), negative_slope=0.1)

        return x


class up2(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(up2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, outChannels, 3, 1, padding=1), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(inChannels, outChannels, 1, padding=1)
        self.conv = nn.Sequential(nn.Conv2d(outChannels, outChannels, 1, 1, padding=0), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels, outChannels, 3, 1, 1)
        self.conv_out = nn.Sequential(nn.Conv2d(outChannels*2, outChannels, 3, 1, padding=1), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels*3, outChannels, 3, 1, 1)
        # self.att = Attention(inChannels, num_heads, bias=True)
        # self.bn = nn.BatchNorm2d(inChannels)

    def forward(self, x, skpCn1):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        # x = self.bn(self.att(x))

        x_1 = self.conv1(x)
        x = F.leaky_relu(x_1, negative_slope=0.1)
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        x = torch.cat((x, skpCn1), 1)

        x = F.leaky_relu(self.conv_out(x), negative_slope=0.1)

        return x


class down(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(down, self).__init__()
        self.relu1 = nn.LeakyReLU(0.1, True)
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1)
        self.norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.bn(x)
        x = self.relu1(x)
        return x


class up_light(nn.Module):
    def __init__(self, inChannels, outChannels, norm=False):
        super(up_light, self).__init__()
        bias = False if norm else True
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(outChannels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(outChannels, track_running_stats=True)
        elif norm == False:
            print('No Normalization.')
        else:
            raise ValueError("Choose BN or IN or False.")

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv1(torch.cat((x, skpCn), 1))
        if self.norm:
            x = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


class down_light(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(down_light, self).__init__()
        self.relu1 = nn.LeakyReLU(0.1, True)
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1)
        self.norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.bn(x)
        x = self.relu1(x)
        return x


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.sigmoid(x)


class CondConv2D(_ConvNd):
    r"""Learn specialized convolutional kernels for each example.
    As described in the paper
    `CondConv: Conditionally Parameterized Convolutions for Efficient Inference`_ ,
    conditionally parameterized convolutions (CondConv),
    which challenge the paradigm of static convolutional kernels
    by computing convolutional kernels as a function of the input.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts per layer
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    .. _CondConv: Conditionally Parameterized Convolutions for Efficient Inference:
       https://arxiv.org/abs/1904.04971
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=4, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)


class Metric(object):
    def __init__(self):
        self.L1Loss_this_epoch = []
        self.Lpips_this_epoch = []
        self.FeaLoss_this_epoch = []
        self.total_this_epoch = []

        self.L1Loss_history = []
        self.Lpips_history = []
        self.FeaLoss_history = []
        self.total_history = []

    def update(self, L1Loss, Lpips, FeaLoss, total):
        self.L1Loss_this_epoch.append(L1Loss.item())
        self.Lpips_this_epoch.append(Lpips.item())
        self.FeaLoss_this_epoch.append(FeaLoss.item())
        self.total_this_epoch.append(total.item())

    def update_epoch(self):
        avg = self.get_average_epoch()
        self.L1Loss_history.append(avg[0])
        self.Lpips_history.append(avg[1])
        self.FeaLoss_history.append(avg[2])
        self.total_history.append(avg[3])
        self.new_epoch()

    def new_epoch(self):
        self.L1Loss_this_epoch = []
        self.Lpips_this_epoch = []
        self.FeaLoss_this_epoch = []
        self.total_this_epoch = []

    def get_average_epoch(self):
        return np.average(self.L1Loss_this_epoch), np.average(self.Lpips_this_epoch), np.average(self.FeaLoss_this_epoch), np.average(self.total_this_epoch)