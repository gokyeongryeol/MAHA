import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
class ImageEmbedding(nn.Module):
    def __init__(self, dim_hidden, dp_rate, is_pretrain):
        super(ImageEmbedding, self).__init__()
        if is_pretrain:
            self.conv_embedding = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, padding=2),
                                                nn.ReLU(),
                                                nn.ZeroPad2d((0,1,0,1)),
                                                nn.MaxPool2d(kernel_size=3, stride=2),
                                                nn.LocalResponseNorm(size=4),
                                                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                                                nn.ReLU(),
                                                nn.LocalResponseNorm(size=4),
                                                nn.ZeroPad2d((0,1,0,1)),
                                                nn.MaxPool2d(kernel_size=3, stride=2))
        
            self.linear_embedding = nn.Sequential(nn.Dropout(p=dp_rate),
                                                  nn.Linear(32*21*21, 384),
                                                  nn.ReLU(),
                                                  nn.Linear(384, dim_hidden))
        
        else:
            self.conv_embedding = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(),
                                                nn.ZeroPad2d((0,1,0,1)),
                                                nn.MaxPool2d(kernel_size=3, stride=2),
                                                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(),
                                                nn.ZeroPad2d((0,1,0,1)),
                                                nn.MaxPool2d(kernel_size=3, stride=2),
                                                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(),
                                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(),
                                                nn.ZeroPad2d((0,1,0,1)),
                                                nn.MaxPool2d(kernel_size=3, stride=2))
        
            self.linear_embedding = nn.Sequential(nn.Dropout(p=dp_rate),
                                                  nn.Linear(32*5*5, dim_hidden),
                                                  nn.ReLU(),
                                                  nn.Linear(dim_hidden, dim_hidden))
            
        
    def forward(self, image):
        fin_emb = []
        for batch_idx in range(len(image)):
            P,H,W,C = image[batch_idx].size()
            conv_emb_idx = self.conv_embedding(image[batch_idx].permute(0,3,1,2))
            flatten_idx = conv_emb_idx.reshape(P,-1)
            fin_emb_idx = self.linear_embedding(flatten_idx)
            fin_emb.append(fin_emb_idx)            
        return fin_emb


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_fn, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, bn_fn):
        super(ResNet, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = bn_fn(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0], bn_fn)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], bn_fn, stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], bn_fn, stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], bn_fn, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, bn_fn, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bn_fn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_fn, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_fn))

        return nn.Sequential(*layers)

    def forward(self, x, param_dict=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet18(pretrained=True, pretrained_model_path='../models/resnet18/pretrained_resnet.pt.tar', **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    nl = nn.BatchNorm2d

    model = ResNet(BasicBlock, [2, 2, 2, 2], nl, **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])
    return model