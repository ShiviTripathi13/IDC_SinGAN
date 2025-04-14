import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=3, nfc=64, num_layers=5, ker_size=3, padd_size=1):
        super(Generator, self).__init__()
        layers = []
        # First layer: from 3 channels to nfc filters
        layers.append(nn.Conv2d(in_channels, nfc, kernel_size=ker_size, padding=padd_size))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        # Intermediate layers
        for i in range(num_layers - 2):
            layers.append(nn.Conv2d(nfc, nfc, kernel_size=ker_size, padding=padd_size))
            layers.append(nn.BatchNorm2d(nfc))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        # Final layer: map back to 3 channels with Tanh activation
        layers.append(nn.Conv2d(nfc, in_channels, kernel_size=ker_size, padding=padd_size))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, nfc=64, num_layers=5, ker_size=3, padd_size=1):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, nfc, kernel_size=ker_size, padding=padd_size))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        for i in range(num_layers - 2):
            layers.append(nn.Conv2d(nfc, nfc, kernel_size=ker_size, padding=padd_size))
            layers.append(nn.BatchNorm2d(nfc))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        layers.append(nn.Conv2d(nfc, 1, kernel_size=ker_size, padding=padd_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
