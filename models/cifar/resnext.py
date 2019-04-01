from __future__ import division
""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
import from https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math
import torch.utils.hooks as hooks
from torch.nn.parameter import Parameter

__all__ = ['resnext']

class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, in_channels, out_channels, stride, cardinality, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * out_channels // widen_factor
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


# -------------------------------------------------------------------------------------------
class WeightNorm(object):
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return v * (1. / self.norm(v))

    def norm(self, p):
        """Computes the norm over all dimensions except dim"""
        if self.dim is None:
            return p.norm()
        if self.dim != 0:
            p = p.transpose(0, self.dim)
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        p = p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
        if self.dim != 0:
            p = p.transpose(0, self.dim)
        return p

    @staticmethod
    def apply(module, name, dim):
        fn = WeightNorm(name, dim)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(fn.norm(weight).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        handle = hooks.RemovableHandle(module._forward_pre_hooks)
        module._forward_pre_hooks[handle.id] = fn
        fn.handle = handle

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)

        self.handle.remove()
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module, name='weight', dim=0):
    """Applies weight normalization to a parameter in the given module.
    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}
    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.
    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.
    See https://arxiv.org/abs/1602.07868
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm
    Returns:
        The original module with the weight norm hook
    Example::
        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])
    """
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module, name='weight'):
    """Removes the weight normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
# -------------------------------------------------------------------------------------------


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, cardinality, depth, num_classes, widen_factor=4, dropRate=0):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            num_classes: number of classes
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        # self.classifier = nn.Linear(1024, num_classes)
        self.classifier = weight_norm(nn.Linear(1024, num_classes, False), name='weight', dim=0)

        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, 1024)
        return self.classifier(x)

def resnext(**kwargs):
    """Constructs a ResNeXt.
    """
    model = CifarResNeXt(**kwargs)
    return model