"""
Train a conditional GAN on MNIST.

Source: https://github.com/arturml/mnist-cgan/blob/master/mnist-cgan.ipynb
"""""
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision import transforms

from cgan import Generator
from cgan import Discriminator
from cgan import generate_digits


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()

    fake_images, fake_labels = generate_digits(generator, batch_size)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, torch.ones(batch_size).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, torch.ones(batch_size).cuda())

    # train with fake images
    fake_images, fake_labels = generate_digits(generator, batch_size)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size).cuda())

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train():
    num_epochs = 50
    display_step = 50
    batch_size = 32

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    data_loader = torch.utils.data.DataLoader(
        MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch), end=' ')
        for i, (images, labels) in enumerate(data_loader):

            step = epoch * len(data_loader) + i + 1
            real_images = images.cuda()
            labels = labels.cuda()
            generator.train()

            d_loss = discriminator_train_step(len(real_images), discriminator,
                                              generator, d_optimizer, criterion,
                                              real_images, labels)

            g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)

            writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)

            if step % display_step == 0:
                generator.eval()
                z = torch.randn(9, 100).cuda()
                labels = torch.LongTensor(np.arange(9)).cuda()
                sample_images = generator(z, labels).unsqueeze(1)
                grid = make_grid(sample_images, nrow=3, normalize=True)
                writer.add_image('sample_image', grid, step)
        print('Done!')

    torch.save(generator.state_dict(), 'generator_state.pt')


if __name__ == "__main__":
    train()
