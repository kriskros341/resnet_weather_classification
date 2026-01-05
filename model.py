import torch
import torch.nn as nn
import torch.nn.functional as F


# AI Pomogło
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First Convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second Convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (Identity mapping)
        self.shortcut = nn.Sequential()
        
        # If input/output dimensions don't match, we must transform the identity
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        
        # The core Residual logic: Element-wise addition
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm2d(128)
        self.res_layer_1 = self._make_layer(128, 256, stride=2)
        self.res_layer_2 = self._make_layer(256, 512, stride=1)
        # self.res_layer_3 = self._make_layer(512, 1024, stride=1)


        # To sprawia, że niezależnie od rozmiaru wejściowego obrazu, wyjście z warstw konwolucyjnych będzie miało rozmiar 1x1 przed wejściem do warstwy w pełni połączonej - pytorch sam dobierze parametry poolowania by to zapewnić
        # Utracone informacje dotyczą umieszczenia features w obrazie, ale dla klasyfikacji całych obrazów nie ma to większego znaczenia
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # mapowanie rezultatów na klasy
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # First convolution - prepare for residual layers
        out = self.conv1(x)
        # out = self.pool(out)
        out = self.bn1(out)
        out = F.relu(out)
        # print(out.size())

        # # Residual layers
        out = self.res_layer_1(out)
        out = self.res_layer_2(out)
        # out = self.res_layer_3(out)
        # print(out.size())
        # # Output from residual layers to fully connected layer
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.drop(out)
        out = self.fc(out)
        return out

