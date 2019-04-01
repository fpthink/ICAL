'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn


__all__ = ['largemargin']


class LargeMargin(nn.Module):

    def __init__(self, num_classes=10):
        super(LargeMargin, self).__init__()
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
        self.bn_fc0 = nn.BatchNorm2d(num_fc)


        self.fc = nn.Linear(num_fc, num_classes)
        
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
        x = self.relu(self.bn_fc0(x))

        # print(x.size())

        x = self.fc(x)
        # print(x.size())

        return x


def largemargin(**kwargs):
    model = LargeMargin(**kwargs)
    return model
