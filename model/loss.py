import torch
from torch import nn
import torchvision.models.vgg as vgg


class LossNetwork(nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg16_bn(pretrained=True).features
        self.layer_name_mapping = {'5': "relu1", '12': "relu2", '22': "relu3", '32': "relu4"}

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        relu1 = output["relu1"]
        relu2 = output["relu2"]
        relu3 = output["relu3"]
        relu4 = output["relu4"]
        return relu1, relu2, relu3, relu4


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.net = LossNetwork().cuda()
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

    def CompareLoss(self, output, gt, neg_sample):

        # bs, C, H, W = neg_sample.shape
        # neg_sample = neg_sample.reshape(bs, -1, 3, H, W)
        # neg_sample = torch.mean(neg_sample, dim=1)

        # neg_sample0 = torch.zeros([bs, H, W])
        # for k in range(int(C/3)):
        #     neg_sample0 += neg_sample[:, k*3:k*3+3, ...]
        # neg_sample0 /= int(C/3)

        output_fea = self.net(output)
        pos_fea = self.net(gt)
        neg_fea = self.net(neg_sample)

        loss_cmp = 0

        weight1 = [3/4, 1/4]

        for i in range(2):
            loss_cmp += weight1[i] * self.L1(output_fea[i], pos_fea[i]) / self.L1(output_fea[i], neg_fea[i])
            # loss_cmp += weight1[i] * self.L1(output_fea[i], neg_fea[i])

        return loss_cmp
        # return pos_fea, neg_fea