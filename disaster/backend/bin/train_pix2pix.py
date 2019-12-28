import argparse
import json
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
from uuid import uuid4

from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from shapely import wkt
from torch import nn
from torch import optim
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop as torchvision_crop

from pix2pix import Discriminator
from pix2pix import Generator


TRAIN_BATCH_SIZE = 1
HEIGHT = 512
WIDTH = 512
TEST_BATCH_SIZE = 1
GENERATOR_FILTERS = 64
DISCRIMINATOR_FILTERS = 64
TRAIN_EPOCHS = 200
GENERATOR_LR = 0.0002
DISCRIMINATOR_LR = 0.0002
L1_LAMBDA = 100
BETA_1 = 0.5
BETA_2 = 0.999


class XView2Dataset(Dataset):
    DAMAGE_COLOUR_MAPPING = {
        "no-damage": (30, 30, 30),
        "minor-damage": (75, 75, 75),
        "major-damage": (150, 150, 150),
        "destroyed": (225, 225, 225),
        "un-classified": (255, 255, 255)
    }
    RAW_HEIGHT, RAW_WIDTH = 1024, 1024

    def __init__(self, directory: Path, regenerate_cache: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._directory = directory

        self._regenerate_cache = regenerate_cache
        self._samples = [
            {
                "image_path": path,
                "label_path": self.label_path(path),
                "label_image_path": self.label_image_path(path)
            }
            for path in list((directory / Path("images")).glob("*.png"))
        ]
        self._transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        sample = self._samples[item]

        image_path = sample["image_path"]
        label_image_path = sample["label_image_path"]

        if not label_image_path.is_file() or self._regenerate_cache:
            self.cache_label_image(image_path)

        y = Image.open(image_path)
        x = Image.open(label_image_path)

        # Apply the same random crop to both the images
        i, j, h, w = transforms.RandomCrop.get_params(
            x, output_size=(HEIGHT, WIDTH)
        )
        x = torchvision_crop(x, i, j, h, w)
        y = torchvision_crop(y, i, j, h, w)

        return self._transforms(x), self._transforms(y)

    def __len__(self):
        return len(self._samples)

    def cache_label_image(self, image_path: Path):
        with self.label_path(image_path).open() as f:
            label_data = json.load(f)

        label_image = self.create_label_image(label_data)

        label_image.save(self.label_image_path(image_path))

    @staticmethod
    def create_label_image(label_data: dict) -> Image:
        label_image = Image.new('RGB', (XView2Dataset.RAW_HEIGHT, XView2Dataset.RAW_WIDTH), 0)

        for building in label_data["features"]["xy"]:
            x, y = wkt.loads(building["wkt"]).exterior.coords.xy
            p = list(zip(x, y))
            try:
                colour = XView2Dataset.DAMAGE_COLOUR_MAPPING[building["properties"]["subtype"]]
            except KeyError:
                # In the case that the building has no 'subtype' property the
                # building is not damaged
                colour = XView2Dataset.DAMAGE_COLOUR_MAPPING["no-damage"]

            ImageDraw.Draw(label_image).polygon(p, colour)

        return label_image

    @staticmethod
    def label_path(p: Path) -> Path:
        return Path(str(p).replace("image", "label").replace(".png", ".json"))

    @staticmethod
    def label_image_path(p: Path) -> Path:
        return Path(str(p).replace("image", "label"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", type=Path)
    parser.add_argument(
        "--regenerate_cache",
        action="store_true",
        help="Whether to force regenerate of label image cache."
    )
    args = parser.parse_args()

    dataset = XView2Dataset(
        args.data_directory,
        args.regenerate_cache
    )
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 10, 10])

    # Create a dev train dataset with just 10 samples
    #train_dataset, _ = torch.utils.data.random_split(train_dataset, [10, len(train_dataset) - 10])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE
    )

    generator = Generator(GENERATOR_FILTERS)
    discriminator = Discriminator(DISCRIMINATOR_FILTERS)

    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)
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
