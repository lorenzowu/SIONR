import torch.nn as nn
import torch
import torch.nn.init as init


def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class SIONR(nn.Module):
    def __init__(self, inplace=True):
        super(SIONR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=inplace),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=inplace),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=inplace),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=inplace),
        )

        self.high = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU(inplace=inplace),
        )

        self.spatial = nn.Sequential(
            nn.Linear(in_features=64, out_features=1),
            nn.LeakyReLU(inplace=inplace),
        )

        self.temporal = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(inplace=inplace),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 + 128, out_features=64),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=64, out_features=1),
            nn.LeakyReLU(inplace=inplace),
        )

        self.weight_init()

    def weight_init(self):
        initializer = kaiming_init
        for block in self._modules:
            for m in self._modules[block]:
                    initializer(m)

    def forward(self, video, feature):
        # batch_size, channel, depth, height, width
        out_tensor = self.conv1(video)
        out_tensor = self.conv2(out_tensor)
        out_tensor = self.conv3(out_tensor)
        out_tensor = self.conv4(out_tensor)

        # low-level temporal variation
        diff_tensor = torch.abs(out_tensor[:, :, 0::2, :, :] - out_tensor[:, :, 1::2, :, :])

        # temporal factor
        out_feature1 = torch.mean(diff_tensor, dim=[3, 4])
        # spatial factor
        out_feature2 = torch.mean(out_tensor[:, :, 1::2, :, :], dim=[3, 4])

        # batch_size, channel, depth
        out_feature1 = out_feature1.permute([0, 2, 1])
        out_feature2 = out_feature2.permute([0, 2, 1])

        # spatiotemporal feature fusion
        out_feature_L = self.temporal(out_feature1) * self.spatial(out_feature2)

        # high-level temporal variation
        feature_abs = torch.abs(feature[:, 0::2] - feature[:, 1::2])
        out_feature_H = self.high(feature_abs)

        # hierarchical feature fusion
        score = self.fc(torch.cat((out_feature_L, out_feature_H), dim=2))

        # mean pooling
        score = torch.mean(score, dim=[1, 2])

        return score
