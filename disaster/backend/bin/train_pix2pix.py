import argparse
from pathlib import Path
from uuid import uuid4

from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import structlog
from torch import nn
from torch import optim
import torch
from torch.utils.data import Dataset

from pix2pix import Discriminator
from pix2pix import Generator
from data import XView2Dataset

LOGGER = structlog.get_logger()

TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
GENERATOR_FILTERS = 64
DISCRIMINATOR_FILTERS = 64
TRAIN_EPOCHS = 200
GENERATOR_LR = 0.0002
DISCRIMINATOR_LR = 0.0002
L1_LAMBDA = 100
BETA_1 = 0.5
BETA_2 = 0.999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--generator-weights", type=Path)
    parser.add_argument("--discriminator-weights", type=Path)

    args = parser.parse_args()

    generator = Generator(GENERATOR_FILTERS)
    if args.generator_weights is not None:
        LOGGER.info(f"Loading generator weights: {args.generator_weights}")
        generator.load_state_dict(torch.load(args.generator_weights))
    else:
        generator.weight_init(mean=0.0, std=0.02)

    discriminator = Discriminator(DISCRIMINATOR_FILTERS)
    if args.discriminator_weights is not None:
        LOGGER.info(f"Loading discriminator weights: {args.discriminator_weights}")
        discriminator.load_state_dict(torch.load(args.discriminator_weights))
    else:
        discriminator.weight_init(mean=0.0, std=0.02)

    dataset = XView2Dataset(
        args.data_directory,
    )
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 10, 10])

    # Create a dev train dataset with just 10 samples
    # train_dataset, _ = torch.utils.data.random_split(train_dataset, [10, len(train_dataset) - 10])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE
    )

    generator.cuda()
    discriminator.cuda()

    generator.train()
    discriminator.train()

    BCE_loss = nn.BCELoss().cuda()
    L1_loss = nn.L1Loss().cuda()

    generator_optimizer = optim.Adam(generator.parameters(), lr=GENERATOR_LR, betas=(BETA_1, BETA_2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(BETA_1, BETA_2))

    def step(engine, batch):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        discriminator.zero_grad()
        discriminator_result = discriminator(x, y).squeeze()
        discriminator_real_loss = BCE_loss(discriminator_result, torch.ones(discriminator_result.size()).cuda())

        generator_result = generator(x)
        discriminator_result = discriminator(x, generator_result).squeeze()

        discriminator_fake_loss = BCE_loss(discriminator_result, torch.zeros(discriminator_result.size()).cuda())
        discriminator_train_loss = (discriminator_real_loss + discriminator_fake_loss) * 0.5
        discriminator_train_loss.backward()
        discriminator_optimizer.step()

        generator.zero_grad()
        generator_result = generator(x)
        # TODO Work out if the below time saving technique impacts training.
        #generator_result = generator_result.detach()
        discriminator_result = discriminator(x, generator_result).squeeze()

        l1_loss = L1_loss(generator_result, y)
        bce_loss = BCE_loss(discriminator_result, torch.ones(discriminator_result.size()).cuda())

        G_train_loss = bce_loss + L1_LAMBDA * l1_loss
        G_train_loss.backward()
        generator_optimizer.step()

        return {
            'generator_train_loss': G_train_loss.item(),
            'discriminator_real_loss': discriminator_real_loss.item(),
            'discriminator_fake_loss': discriminator_fake_loss.item(),
        }

    trainer = Engine(step)

    tb_logger = TensorboardLogger(log_dir=f"tensorboard/logdir/{uuid4()}")
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training",
            output_transform=lambda out: out,
            metric_names='all'
        ),
        event_name=Events.ITERATION_COMPLETED
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def add_generated_images(engine):
        def min_max(image):
            return (image - image.min()) / (image.max() - image.min())

        for idx, (x, y) in enumerate(test_loader):
            generated = min_max(generator(x.cuda()).squeeze().cpu())
            real = min_max(y.squeeze())

            tb_logger.writer.add_image(
                f"generated_test_image_{idx}",
                # Concatenate the images into a single tiled image
                torch.cat([x.squeeze(), generated, real], 2),
                global_step=engine.state.epoch
            )

    checkpoint_handler = ModelCheckpoint(
        "checkpoints/", "pix2pix",
        n_saved=1, require_empty=False, save_interval=1
    )
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={
            'generator': generator,
            'discriminator': discriminator
        })

    timer = Timer(average=True)
    timer.attach(
        trainer,
        resume=Events.ITERATION_STARTED,
        step=Events.ITERATION_COMPLETED
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        print(
            "Epoch[{}] Iteration[{}] Duration[{}] Losses: {}".format(
                engine.state.epoch,
                engine.state.iteration,
                timer.value(),
                engine.state.output
            )
        )

    trainer.run(train_loader, max_epochs=TRAIN_EPOCHS)

    tb_logger.close()


if __name__ == "__main__":
    main()
