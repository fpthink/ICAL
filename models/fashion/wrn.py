import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.hooks as hooks
from torch.nn.parameter import Parameter

__all__ = ['wrn']

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

# -------------------------------------------------------------------------------------------
class WeightNorm(object):
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        # print(v.size())
        # print(self.norm(v).size())
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

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        # -----channel 3 modify to 1
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)


        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.fc = weight_norm(nn.Linear(nChannels[3], num_classes, False), name='weight', dim=0)

        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model
