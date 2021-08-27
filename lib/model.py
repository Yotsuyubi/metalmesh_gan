import torch as th
import torch.nn as nn


class NNSimulator(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 256, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(8208, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64*35),
            nn.ReLU(),
        )

        def out_factory(index):
            layers = [
                nn.ConvTranspose1d(64, 32, 15, stride=2, padding=7),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 15, stride=2, padding=7),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 8, 15, stride=2, padding=7),
                nn.ReLU(),
                nn.Conv1d(8, 1, 24),
            ]
            if index <= 5:
                pass
            return nn.Sequential(*layers)

        self.out = nn.ModuleList(
            [
                out_factory(index)
                for index in range(8)
            ]
        )

    def forward(self, x, thickness):
        feature = th.cat([self.feature(x), thickness], dim=1)
        fc = self.fc(feature).view(-1, 64, 35)
        return th.cat([out(fc) for out in self.out], dim=1)


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(8, 64, 15, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 128, 15, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(128, 256, 15, stride=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Conv1d(256, 512, 15, stride=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Flatten()
        )

        self.decoder_img = nn.ModuleList([
            nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(96//4, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256 + 24, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(128 + 6, 128, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid(),
            ),
        ])

        self.decoder_thickness = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode_img = self.decoder_img[0](encode.view(-1, 96, 4, 4))
        decode_img = self.decoder_img[1](
            th.cat([encode.view(-1, 24, 8, 8), decode_img], dim=1)
        )
        decode_img = self.decoder_img[2](
            th.cat([encode.view(-1, 6, 16, 16), decode_img], dim=1)
        )
        decode_thickness = self.decoder_thickness(encode)
        return decode_img, decode_thickness


class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, 5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.encoder(x))


if __name__ == "__main__":

    model = Generator()
    dummy = th.randn(1, 8, 250)
    img, thickness = model(dummy)
    print(img.shape, thickness.shape)

    critic = Critic()
    y = critic(img)
    print(y.shape)

    nn_sim = NNSimulator()
    print(nn_sim(img, thickness).shape)
