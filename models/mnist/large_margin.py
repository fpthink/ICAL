'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn


__all__ = ['largemargin']


class LargeMargin(nn.Module):

    def __init__(self, num_classes=10):
        super(LargeMargin, self).__init__()
        
        # conv0
        self.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) # Conv1.1
        self.bn0 = nn.BatchNorm2d(64)
        

        # conv1
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv1.1
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv1.2
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv1.3
        self.bn1_3 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv2
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv2.1
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv2.2
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv2.3
        self.bn2_3 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv3
        
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv3.1
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv3.2
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Conv3.3
        self.bn3_3 = nn.BatchNorm2d(64)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc0 = nn.Linear(576, 256)
        self.bn_fc0 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # conv0
        x = self.relu(self.bn0(self.conv0(x)))

        # conv1.x
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.relu(self.bn1_3(self.conv1_3(x)))
        
        # print('conv1', x.size())
        x = self.pool1(x)
        # print('pool1', x.size())

        # conv2.x
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.relu(self.bn2_3(self.conv2_3(x)))
        
        # print('conv2', x.size())
        x = self.pool2(x)
        # print('pool2', x.size())

        # conv3.x
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        
        # print('conv3', x.size())
        x = self.pool3(x)
        # print('pool3', x.size())

        x = x.view(x.size(0), -1)
        # print(x.size())

        x = self.relu(self.bn_fc0(self.fc0(x)))

        # print(x.size())

        x = self.fc(x)
        # print(x.size())
        # exit()

        return x


def largemargin(**kwargs):
    model = LargeMargin(**kwargs)
    return model
