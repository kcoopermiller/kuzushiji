import torch
import torch.nn as nn

# TODO: Rudimentary implementation of Kuronet

class KuroNet(nn.Module):
    def __init__(self, num_experts=4):
        super(KuroNet, self).__init__()

        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.center = ResidualBlock(512, 512)

        self.decoder4 = DecoderBlock(1024, 256)
        self.decoder3 = DecoderBlock(512, 128)
        self.decoder2 = DecoderBlock(256, 64)
        self.decoder1 = DecoderBlock(128, 64)

    def forward(self, x):
        enc1, x = self.encoder1(x)
        enc2, x = self.encoder2(x)
        enc3, x = self.encoder3(x)
        enc4, x = self.encoder4(x)

        x = self.center(x)

        x = self.decoder4(x, enc4)
        x = self.decoder3(x, enc3)
        x = self.decoder2(x, enc2)
        x = self.decoder1(x, enc1)

        return x    

# Residual Fusion Network Building Blocks:
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=16):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.resblock = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.resblock(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.resblock = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # Skip connection
        x = self.resblock(x)
        return x

# TODO: Mixture of Experts Network Building Blocks:

class Expert(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=128):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            EncoderBlock(1, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 512),
            ResidualBlock(512, 512),
            DecoderBlock(1024, 256),
            DecoderBlock(512, 128),
            DecoderBlock(256, 64),
            DecoderBlock(128, 64)
        )

    def forward(self, x):
        return self.network(x)

class Gate(nn.Module):
    def __init__(self, in_features, num_experts):
        super(Gate, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)
    
class MoE(nn.Module):
    def __init__(self, in_features, out_features, num_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(in_features, out_features) for _ in range(num_experts)])
        self.gate = Gate(in_features, num_experts)

    def forward(self, x):
        # Flatten input if necessary
        original_shape = x.shape
        x = x.view(x.size(0), -1)

        gating_weights = self.gate(x)  # Get weights for each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, out_features]

        # Weighted sum of expert outputs
        expanded_gates = gating_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        output = torch.sum(expert_outputs * expanded_gates, dim=1)  # [batch_size, out_features]

        # Reshape output to match original input shape
        if len(original_shape) > 2:
            output = output.view(original_shape[0], original_shape[1], -1)

        return output