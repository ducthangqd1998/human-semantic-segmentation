# #------------------------------------------------------------------------------
# #  Libraries
# #------------------------------------------------------------------------------
# from models.base import BaseBackboneWrapper
# from timm.models.resnet import ResNet as BaseResNet
# from timm.models.resnet import default_cfgs, load_pretrained, BasicBlock, Bottleneck

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import OrderedDict


# #------------------------------------------------------------------------------
# #   ResNetBlock
# #------------------------------------------------------------------------------
# class ResNetBasicBlock(nn.Module):
# 	expansion = 1

# 	def __init__(self, in_channels, out_channels):
# 		super(ResNetBasicBlock, self).__init__()
# 		downsample = nn.Sequential(OrderedDict([
# 			("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
# 			("bn", nn.BatchNorm2d(out_channels))
# 		]))
# 		self.block = BasicBlock(
# 			in_channels,
# 			int(out_channels/BasicBlock.expansion),
# 			downsample=downsample,
# 		)

# 	def forward(self, x):
# 		x = self.block(x)
# 		return x


# class ResNetBottleneckBlock(nn.Module):
# 	expansion = 4
	
# 	def __init__(self, in_channels, out_channels):
# 		super(ResNetBottleneckBlock, self).__init__()
# 		downsample = nn.Sequential(OrderedDict([
# 			("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
# 			("bn", nn.BatchNorm2d(out_channels))
# 		]))
# 		self.block = Bottleneck(
# 			in_channels,
# 			int(out_channels/Bottleneck.expansion),
# 			downsample=downsample,
# 		)

# 	def forward(self, x):
# 		x = self.block(x)
# 		return x


# #------------------------------------------------------------------------------
# #  ResNet
# #------------------------------------------------------------------------------
# class ResNet(BaseResNet, BaseBackboneWrapper):
# 	def __init__(self, block, layers, frozen_stages=-1, norm_eval=False, **kargs):
# 		super(ResNet, self).__init__(block=block, layers=layers, **kargs)
# 		self.frozen_stages = frozen_stages
# 		self.norm_eval = norm_eval

# 	def forward(self, input):
# 		# Stem
# 		x1 = self.conv1(input)
# 		x1 = self.bn1(x1)
# 		x1 = self.relu(x1)
# 		# Stage1
# 		x2 = self.maxpool(x1)
# 		x2 = self.layer1(x2)
# 		# Stage2
# 		x3 = self.layer2(x2)
# 		# Stage3
# 		x4 = self.layer3(x3)
# 		# Stage4
# 		x5 = self.layer4(x4)
# 		# Output
# 		return x1, x2, x3, x4, x5

# 	def init_from_imagenet(self, archname):
# 		load_pretrained(self, default_cfgs[archname], self.num_classes)

# 	def _freeze_stages(self):
# 		# Freeze stem
# 		if self.frozen_stages>=0:
# 			self.bn1.eval()
# 			for module in [self.conv1, self.bn1]:
# 				for param in module.parameters():
# 					param.requires_grad = False

# 		# Chosen subsequent blocks are also frozen
# 		for stage_idx in range(1, self.frozen_stages+1):
# 			for module in getattr(self, "layer%d"%(stage_idx)):
# 				module.eval()
# 				for param in module.parameters():
# 					param.requires_grad = False


# #------------------------------------------------------------------------------
# #  Versions of ResNet
# #------------------------------------------------------------------------------
# def resnet18(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
# 	"""Constructs a ResNet-18 model.
# 	"""
# 	default_cfg = default_cfgs['resnet18']
# 	model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, **kwargs)
# 	model.default_cfg = default_cfg
# 	if pretrained:
# 		load_pretrained(model, default_cfg, num_classes, in_chans)
# 	return model

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, embedding_size=64):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_embed = nn.Linear(256 * block.expansion, embedding_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_embed(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2], **kwargs)
    if pretrained:
        state = model.state_dict()
        loaded_state_dict = model_zoo.load_url(model_urls['resnet18'])
        for k in loaded_state_dict:
            if k in state:
                state[k] = loaded_state_dict[k]
        model.load_state_dict(state)
    return model
