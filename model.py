import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, in_channels=24):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.model(x)

        # 前半の256次元を平均、後半の256次元を分散として扱う
        mean, logvar = torch.split(x, 256, dim=1)

        return mean, logvar


class ConditionalBatchNorm1d(nn.Module):

    def __init__(self, in_channel, n_condition):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_channel, affine=False)

        self.embedding = nn.Embedding(n_condition, in_channel * 2)
        self.embedding.weight.data[:, :in_channel] = 1
        self.embedding.weight.data[:, in_channel:] = 0

    def forward(self, x, y):
        x = self.bn(x)

        emb = self.embedding(y)
        mean, std = emb.chunk(2, 1)
        mean = mean.unsqueeze(2)
        std = std.unsqueeze(2)

        x = mean * x + std

        return x


class DecoderBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 n_speaker,
                 upsample_rate=1):
        super().__init__()

        self.is_up = upsample_rate > 1
        if upsample_rate > 1:
            self.up = nn.Upsample(scale_factor=upsample_rate, mode='linear', align_corners=True)

        self.is_linear = in_channels != 256
        if in_channels != 256:
            self.fc1 = nn.Linear(256, in_channels)

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        #self.cbn = ConditionalBatchNorm1d(out_channels, n_speaker)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        if self.is_up:
            x = self.up(x)
        if self.is_linear:
            y = self.fc1(y)

        x = x + y.unsqueeze(-1).expand(x.size())

        x = self.conv(x)
        x = self.bn(x)
        #x = self.cbn(x, y)
        x = self.relu(x)

        return x


class Decoder(nn.Module):

    def __init__(self, out_channels, n_speaker):
        super().__init__()

        self.blocks = nn.ModuleList([
            DecoderBlock(256, 128, 3, 1, 1, n_speaker),
            DecoderBlock(128, 64, 3, 1, 1, n_speaker, 2),
            DecoderBlock(64, 32, 3, 1, 1, n_speaker, 2),
            DecoderBlock(32, 32, 3, 1, 1, n_speaker, 2),
            DecoderBlock(32, 32, 3, 1, 1, n_speaker, 2),
        ])

        self.embedding  = nn.Embedding(n_speaker, 256)

        self.out_conv = nn.Conv1d(32, out_channels, 5, 1, 2)

    def forward(self, x, label):
        y = self.embedding(label)
        for block in self.blocks:
            x = block(x, y)

        x = self.out_conv(x)

        return x


class VAE(nn.Module):

    def __init__(self, mcep_channels, n_speaker):
        super().__init__()

        self.encoder = Encoder(mcep_channels)
        self.decoder = Decoder(mcep_channels, n_speaker)

    def forward(self, x, label):
        mean, logvar = self.encoder(x)

        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mean + eps * std

        x = self.decoder(x, label)

        return x, mean, logvar
