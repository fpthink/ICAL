import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.utils.hooks as hooks
from torch.nn.parameter import Parameter


__all__ = ['densenet']


from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

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

class DenseNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(self.inplanes, num_classes, False)
        self.fc = weight_norm(nn.Linear(self.inplanes, num_classes, False), name='weight', dim=0)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)


    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x)) 
        x = self.trans2(self.dense2(x)) 
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def densenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)
