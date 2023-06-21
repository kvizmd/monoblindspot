"""
Official Implementation of Deep Layer Aggregation

cite: https://github.com/ucbdrive/dla

BSD 3-Clause License

Copyright (c) 2018, Fisher Yu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import math

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo

from .layer import BasicBlock, Bottleneck, BottleneckX, Tree


class DLA(nn.Module):
    def __init__(
            self,
            levels,
            channels,
            block=BasicBlock,
            norm_func=nn.BatchNorm2d,
            residual_root=False,
            linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels

        self.strict = isinstance(norm_func, nn.BatchNorm2d)

        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False),
            norm_func(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
                channels[0],
                channels[0],
                levels[0],
                norm_func=norm_func)
        self.level1 = self._make_conv_level(
                channels[0],
                channels[1],
                levels[1],
                stride=2,
                norm_func=norm_func)
        self.level2 = Tree(
                levels[2],
                block,
                channels[1],
                channels[2],
                2,
                level_root=False,
                root_residual=residual_root,
                norm_func=norm_func)
        self.level3 = Tree(
                levels[3],
                block,
                channels[2],
                channels[3],
                2,
                level_root=True,
                root_residual=residual_root,
                norm_func=norm_func)
        self.level4 = Tree(
                levels[4],
                block,
                channels[3],
                channels[4],
                2,
                level_root=True,
                root_residual=residual_root,
                norm_func=norm_func)
        self.level5 = Tree(
                levels[5],
                block,
                channels[4],
                channels[5],
                2,
                level_root=True,
                root_residual=residual_root,
                norm_func=norm_func)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_func):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(
            self,
            block,
            inplanes,
            planes,
            blocks,
            stride=1,
            norm_func=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                norm_func(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(
            self,
            inplanes,
            planes,
            convs,
            stride=1,
            dilation=1,
            norm_func=nn.BatchNorm2d):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                norm_func(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(
            self,
            name,
            hash,
            data='imagenet'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = os.path.join(
                'http://dl.yf.io/dla/models',
                data,
                '{}-{}.pth'.format(name, hash))

            model_weights = model_zoo.load_url(model_url)

        for key in ['fc.weight', 'fc.bias']:
            model_weights.pop(key)

        self.load_state_dict(model_weights, strict=self.strict)


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla34', 'ba72cf86')
    return model


def dla46_c(pretrained=True, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla46_c', '2bfd52c3')
    return model


def dla46x_c(pretrained=True, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla46x_c', 'd761bae7')
    return model


def dla60x_c(pretrained=True, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla60x_c', 'b870c45c')
    return model


def dla60(pretrained=True, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla60', '24839fc4')
    return model


def dla60x(pretrained=True, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla60x', 'd15cacda')
    return model


def dla102(pretrained=True, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla102', 'd94d9790')
    return model


def dla102x(pretrained=True, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla102x', 'ad62be81')
    return model


def dla102x2(pretrained=True, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla102x2', '262837b6')
    return model


def dla169(pretrained=True, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model('dla169', '0914e092')
    return model
