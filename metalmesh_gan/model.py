import torch as th
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(*[
            nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(32//4, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid(),
            )
        ])

    def forward(self, x):
        return self.decoder(x.view(-1, 32, 4, 4))


class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, 9, padding=4),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 9, padding=4),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.encoder(x))

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


if __name__ == "__main__":

    model = Generator()
    dummy = th.randn(1, 8, 250)
    img, thickness = model(dummy)
    print(img.shape, thickness.shape)

    critic = Critic()
    y = critic(img)
    print(y.shape)
