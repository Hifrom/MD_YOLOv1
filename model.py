import torch
import torch.nn as nn

class YoloTiny (nn.Module):
    def __init__(self):
        super(YoloTiny, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU(0.1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.LeakyReLU(0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU(0.1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU(0.1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU(0.1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU(0.1)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(1024)
        self.relu7 = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(7*7*1024, 500)
        self.dropout = nn.Dropout(0.5)
        self.relu8 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(500, 7 * 7 * (1 + 2 * 5))

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu8(x)
        x = self.fc2(x)
        return x
# model = YoloTiny()
# x = torch.randn(1,3,416,416)
# print(model.forward(x).shape)
