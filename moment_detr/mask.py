import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GumbelSoftmax(nn.Module):
    '''
        gumbel softmax gate.
    '''

    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor + eps) - torch.log(
            1 - uniform_samples_tensor + eps)
        return gumble_samples_tensor

    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits

    def forward(self, logits):
        # if not self.training:
        #    out_hard = (logits>=0).float()
        #    return out_hard, out_hard
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard, out_soft


class Mask_s(nn.Module):
    '''
        Attention Mask spatial.
    '''

    def __init__(self, h, w, planes, block_w, block_h, eps=0.66667,
                 bias=-1, **kwargs):
        super(Mask_s, self).__init__()
        # Parameter
        self.width, self.height, self.channel = w, h, planes
        self.mask_h, self.mask_w = int(np.ceil(h / block_h)), int(np.ceil(w / block_w))
        self.eleNum_s = torch.Tensor([self.mask_h * self.mask_w])
        # spatial attention
        self.atten_s = nn.Conv2d(planes, 1, kernel_size=3, stride=1, bias=bias >= 0, padding=1)
        if bias >= 0:
            nn.init.constant_(self.atten_s.bias, bias)
        # Gate
        self.gate_s = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2, 3))

    def forward(self, x):
        batch, channel, height, width = x.size()
        # Pooling
        input_ds = F.adaptive_avg_pool2d(input=x, output_size=(self.mask_h, self.mask_w))
        # spatial attention
        s_in = self.atten_s(input_ds)  # [N, 1, h, w]
        # spatial gate
        mask_s = self.gate_s(s_in)  # [N, 1, h, w]
        # norm
        norm = self.norm(mask_s)
        norm_t = self.eleNum_s.to(x.device)
        return mask_s, norm, norm_t

    def get_flops(self):
        flops = self.mask_h * self.mask_w * self.channel * 9
        return flops


class Mask_c_old(nn.Module):
    '''
        Attention Mask.
    '''

    def __init__(self, inplanes=256, outplanes=1, fc_reduction=4, eps=0.66667, set_focal_loss_bias=False):
        super(Mask_c_old, self).__init__()
        # Parameter
        self.bottleneck = inplanes // fc_reduction
        self.inplanes, self.outplanes = inplanes, outplanes
        self.eleNum_c = torch.Tensor([outplanes])
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.set_focal_loss_bias = set_focal_loss_bias
        self.atten_c = nn.Sequential(
            nn.Conv1d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(self.bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.bottleneck, outplanes, kernel_size=1, stride=1, bias=True),

        )
        if set_focal_loss_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.atten_c[-1].bias.data = torch.ones(self.atten_c[-1].bias.shape[0]) * bias_value

        # Gate
        self.gate_c = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2))

    def forward(self, x):
        batch, channel, _ = x.size()
        context = self.avg_pool(x)  # [N, C, 1]
        # transform
        c_in = self.atten_c(context)  # [N, C_out, 1]
        # channel gate
        mask_c, prob_soft = self.gate_c(c_in)  # [N, C_out, 1]
        # norm
        # norm = self.norm(mask_c)
        # norm_t = self.eleNum_c.to(x.device)
        return mask_c, prob_soft  # , norm, norm_t

    def get_flops(self):
        flops = self.inplanes * self.bottleneck + self.bottleneck * self.outplanes
        return flops


class Mask_c_2d(nn.Module):
    '''
        Attention Mask.
    '''

    def __init__(self, inplanes, outplanes, fc_reduction=4, eps=0.66667, bias=-1, **kwargs):
        super(Mask_c_2d, self).__init__()
        # Parameter
        self.bottleneck = inplanes // fc_reduction
        self.inplanes, self.outplanes = inplanes, outplanes
        self.eleNum_c = torch.Tensor([outplanes])
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.atten_c = nn.Sequential(
            nn.Conv2d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck, outplanes, kernel_size=1, stride=1, bias=bias >= 0),
        )
        if bias >= 0:
            nn.init.constant_(self.atten_c[3].bias, bias)
        # Gate
        self.gate_c = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2, 3))

    def forward(self, x):
        batch, channel, _, _ = x.size()
        context = self.avg_pool(x)  # [N, C, 1, 1]
        # transform
        c_in = self.atten_c(context)  # [N, C_out, 1, 1]
        # channel gate
        mask_c = self.gate_c(c_in)  # [N, C_out, 1, 1]
        # norm
        norm = self.norm(mask_c)
        norm_t = self.eleNum_c.to(x.device)
        return mask_c, norm, norm_t

    def get_flops(self):
        flops = self.inplanes * self.bottleneck + self.bottleneck * self.outplanes
        return flops


class CNN(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        #TODO
        self.conv1 = nn.Conv1d(in_c, in_c, 3, 1)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(in_c, in_c, 3, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_c, int(in_c // 2))
        self.fc2 = nn.Linear(int(in_c // 2), 1)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.fc2.bias.data = torch.ones(self.fc2.bias.shape[0]) * bias_value

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.avg_pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.squeeze()))
        x = self.fc2(x)
        return x


class Mask_c(nn.Module):
    '''
        Attention Mask.
    '''

    def __init__(self, inplanes=256, eps=0.66667, bias=2,
                 set_focal_loss_bias=False):
        super(Mask_c, self).__init__()

        self.net = CNN(inplanes)
        self.gate_c = GumbelSoftmax(eps=eps)

    def forward(self, x):
        context = self.net(x)  # 4,1
        mask_c, prob_soft = self.gate_c(context)  # [N, C_out]

        return mask_c.unsqueeze(-1), prob_soft
