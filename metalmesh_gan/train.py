import yaml
import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .model import Generator, Critic
from .dataset import GANDataset
import torchvision
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import copy
import argparse


pl.seed_everything(0)
th.manual_seed(0)


class LitGAN(pl.LightningModule):

    def __init__(
        self, num_d_train=5, g_learning_rate=1e-4,
        d_learning_rate=1e-5, **kwargs
    ):
        super().__init__()
        self.generator = Generator()
        self.critic = Critic()
        self.critic_weights = Critic()
        self.save_hyperparameters("g_learning_rate")
        self.save_hyperparameters("d_learning_rate")
        self.save_hyperparameters("num_d_train")

    def forward(self, x):
        image = self.generator(x)
        # make symmetry images.
        image_batch = th.zeros(x.size()[0], 1, 32, 32).type_as(x)
        for i in range(x.size()[0]):
            image_unit = image[i, 0, 32//2:, 32//2:]
            image_tri = th.tril(image_unit)
            image_unit = image_tri + image_tri.T - \
                th.diag(th.diagonal(image_unit))
            image_batch[i, 0, :32//2, :32//2] += th.rot90(image_unit, k=2)
            image_batch[i, 0, :32//2, 32//2:] += th.rot90(image_unit, k=1)
            image_batch[i, 0, 32//2:, 32//2:] += image_unit
            image_batch[i, 0, 32//2:, :32//2] += th.rot90(image_unit, k=-1)
        return image_batch

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch
        z = th.randn(real.size()[0], 512).type_as(real)
        valid = th.ones(real.size(0), 1).type_as(real)
        invalid = th.zeros(real.size(0), 1).type_as(real)

        # train critic.
        if optimizer_idx < self.hparams.num_d_train:

            # store a critic weights after updating in first time for unroll.
            if optimizer_idx == 1:
                self.critic_weights = copy.deepcopy(self.critic)

            real_loss = self.adversarial_loss(
                self.critic(real),
                valid
            )

            fake_loss = self.adversarial_loss(
                self.critic(self(z).detach()),
                invalid
            )

            loss = (real_loss + fake_loss) / 2

            # At good learning, critic_loss goes to 0.3.
            self.log('critic_loss', loss, prog_bar=True)

            return loss

        # train generator.
        if optimizer_idx == self.hparams.num_d_train:

            fake = self(z)
            sample_imgs = fake[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            torchvision.utils.save_image(grid, "test/img.png")

            loss = self.adversarial_loss(
                self.critic(fake),
                valid
            )

            self.log('generator_loss', loss, prog_bar=True)

            self.critic.load(self.critic_weights)  # unroll

            return loss

    def configure_optimizers(self):
        opt_g = th.optim.Adam(
            self.generator.parameters(), lr=self.hparams.g_learning_rate
        )
        opt_d = th.optim.Adam(
            self.critic.parameters(), lr=self.hparams.d_learning_rate
        )
        return [opt_d for _ in range(self.hparams.num_d_train)]+[opt_g], []


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train GANs')
    parser.add_argument('dataset_dir')
    parser.add_argument('--hparams', default="hparams.yml")
    parser.add_argument('--resume', default=None)
    parser.add_argument('--max_epochs', default=999)
    args = parser.parse_args()

    with open(args.hparams, 'r') as yml:
        config = yaml.safe_load(yml)

    batchsize = config["batchsize"]
    num_d_train = config["num_d_train"]
    g_learning_rate = config["g_learning_rate"]
    d_learning_rate = config["d_learning_rate"]

    dataset = GANDataset(root=args.dataset_dir)
    train_loader = DataLoader(dataset, batch_size=batchsize,
                              shuffle=True, num_workers=4)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/', save_last=True,
        monitor="critic_loss"
    )
    gpu = 0
    if th.cuda.is_available():
        gpu = 1
    trainer = pl.Trainer(
        gpus=gpu, callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.resume,
        max_epochs=args.max_epochs, log_every_n_steps=2
    )
    gan = LitGAN(num_d_train, g_learning_rate, d_learning_rate)
    trainer.fit(gan, train_loader)
