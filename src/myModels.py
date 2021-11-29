import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch import nn, optim
from torch.nn import functional as F
import pickle
import argparse
import sys
import os
import PIL

from torchvision.models import vgg16_bn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class VGG16(nn.Module):
    def __init__(self, n_classes=1000, pretrained = True, input_channels = 3):
        super(VGG16, self).__init__()

        self.model = vgg16_bn(num_classes = 1000, pretrained = pretrained)

        self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=n_classes, bias=True)
        )

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def forward(self, x):
        return self.model(x)

    def get_saliency_maps(self, x, ci):
        inp = x.clone()
        inp.requires_grad = True
        self.eval()
        score = self.model(inp)[:,ci].sum()
        score.backward()
        return inp.grad.sum(1)

    def get_CAM(self, x, ci):
        with torch.no_grad():
            pre_gap = self.model.features(x)
            self.eval()
            weights = self.model.classifier[0].weight[ci,:]
            combined = (pre_gap*weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)).sum(1).unsqueeze(1)
            return F.interpolate(combined, x.shape[2], mode = 'bicubic').squeeze()


