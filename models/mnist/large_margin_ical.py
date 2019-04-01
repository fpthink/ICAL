'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn

import torch.utils.hooks as hooks
from torch.nn.parameter import Parameter


__all__ = ['largemarginical']


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

class LargeMarginICAL(nn.Module):

    def __init__(self, num_classes=10):
        super(LargeMarginICAL, self).__init__()
        out0 = 0
        out1 = 0
        out2 = 0
        out3 = 0
        num_fc = 0
        num_c = 0

        if num_classes == 10:
            out0 = 64
            out1 = 64
            out2 = 96
            out3 = 128
            num_fc = 256
            num_c = 2048
        elif num_classes == 100:
            out0 = 96
            out1 = 96
            out2 = 192
            out3 = 384
            num_fc = 512
            num_c = 6144

        # conv0
        self.conv0 = nn.Conv2d(3, out0, kernel_size=3, stride=1, padding=1)   # Conv0
        self.bn0 = nn.BatchNorm2d(out0) 

        # conv1
        self.conv1_1 = nn.Conv2d(out0, out1, kernel_size=3, stride=1, padding=1) # Conv1.1
        self.bn1_1 = nn.BatchNorm2d(out1)
        self.conv1_2 = nn.Conv2d(out1, out1, kernel_size=3, stride=1, padding=1) # Conv1.2
        self.bn1_2 = nn.BatchNorm2d(out1)
        self.conv1_3 = nn.Conv2d(out1, out1, kernel_size=3, stride=1, padding=1) # Conv1.3
        self.bn1_3 = nn.BatchNorm2d(out1)
        self.conv1_4 = nn.Conv2d(out1, out1, kernel_size=3, stride=1, padding=1) # Conv1.4
        self.bn1_4 = nn.BatchNorm2d(out1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv2
        self.conv2_1 = nn.Conv2d(out1, out2, kernel_size=3, stride=1, padding=1) # Conv2.1
        self.bn2_1 = nn.BatchNorm2d(out2)
        self.conv2_2 = nn.Conv2d(out2, out2, kernel_size=3, stride=1, padding=1) # Conv2.2
        self.bn2_2 = nn.BatchNorm2d(out2)
        self.conv2_3 = nn.Conv2d(out2, out2, kernel_size=3, stride=1, padding=1) # Conv2.3
        self.bn2_3 = nn.BatchNorm2d(out2)
        self.conv2_4 = nn.Conv2d(out2, out2, kernel_size=3, stride=1, padding=1) # Conv2.4
        self.bn2_4 = nn.BatchNorm2d(out2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv3
        self.conv3_1 = nn.Conv2d(out2, out3, kernel_size=3, stride=1, padding=1) # Conv3.1
        self.bn3_1 = nn.BatchNorm2d(out3)
        self.conv3_2 = nn.Conv2d(out3, out3, kernel_size=3, stride=1, padding=1) # Conv3.2
        self.bn3_2 = nn.BatchNorm2d(out3)
        self.conv3_3 = nn.Conv2d(out3, out3, kernel_size=3, stride=1, padding=1) # Conv3.3
        self.bn3_3 = nn.BatchNorm2d(out3)
        self.conv3_4 = nn.Conv2d(out3, out3, kernel_size=3, stride=1, padding=1) # Conv3.4
        self.bn3_4 = nn.BatchNorm2d(out3)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc0 = nn.Linear(num_c, num_fc)

        # self.fc = nn.Linear(num_fc, num_classes)
        self.fc = weight_norm(nn.Linear(num_fc, num_classes, False), name='weight', dim=0)

        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # conv0
        x = self.relu(self.bn0(self.conv0(x)))
        # print('conv0', x.size())

        # conv1.x
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.relu(self.bn1_3(self.conv1_3(x)))
        x = self.relu(self.bn1_4(self.conv1_4(x)))

        # print('conv1', x.size())
        x = self.pool1(x)
        # print('pool1', x.size())

        # conv2.x
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.relu(self.bn2_3(self.conv2_3(x)))
        x = self.relu(self.bn2_4(self.conv2_4(x)))

        # print('conv2', x.size())
        x = self.pool2(x)
        # print('pool2', x.size())

        # conv3.x
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.relu(self.bn3_4(self.conv3_4(x)))

        # print('conv3', x.size())
        x = self.pool3(x)
        # print('pool3', x.size())

        x = x.view(x.size(0), -1)
        # print(x.size())

        x = self.fc0(x)
        # print(x.size())

        x = self.fc(x)
        # print(x.size())

        return x


def largemarginical(**kwargs):
    model = LargeMarginICAL(**kwargs)
    return model
