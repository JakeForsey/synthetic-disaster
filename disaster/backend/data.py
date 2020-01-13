import json
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
from typing import Tuple, Optional

import geotiler
from shapely import wkt
from shapely.ops import cascaded_union
import structlog
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop as torchvision_crop

LOGGER = structlog.get_logger()

INPUT_HEIGHT, INPUT_WIDTH = 512, 512
RAW_HEIGHT, RAW_WIDTH = 1024, 1024

DAMAGE_COLOUR_MAPPING = {
    "no-damage": (30, 30, 30),
    "minor-damage": (75, 75, 75),
    "major-damage": (150, 150, 150),
    "destroyed": (225, 225, 225),
    "un-classified": (255, 255, 255)
}
BACKGROUND_COLOUR_MAPPING = {
    'mexico-earthquake': (0, 0, 0, 50),
    'portugal-wildfire': (40, 0, 0, 50),
    'woolsey-fire': (80, 0, 0, 50),
    'pinery-bushfire': (120, 0, 0, 50),
    'santa-rosa-wildfire': (160, 0, 0, 50),
    'socal-fire': (200, 0, 0, 50),
    'hurricane-florence': (240, 0, 0, 50),
    'midwest-flooding': (255, 25, 0, 50),
    'hurricane-harvey': (255, 65, 0, 50),
    'nepal-flooding': (255, 105, 0, 50),
    'sunda-tsunami': (255, 145, 0, 50),
    'palu-tsunami': (255, 185, 0, 50),
    'guatemala-volcano': (255, 225, 0, 50),
    'lower-puna-volcano': (255, 255, 10, 50),
    'hurricane-michael': (255, 255, 50, 50),
    'joplin-tornado': (255, 255, 90, 50),
    'moore-tornado': (255, 255, 130, 50),
    'tuscaloosa-tornado': (255, 255, 170, 50),
    'hurricane-matthew': (255, 255, 210, 50)
}

TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_label_path(p: Path) -> Path:
    return Path(str(p).replace("images", "labels").replace(".png", ".json"))


def get_label_image_path(p: Path) -> Path:
    return Path(str(p).replace("images", "labels"))


def has_buildings(image_path: Path) -> bool:
    with get_label_path(image_path).open() as f:
        label_data = json.load(f)

    return len(label_data["features"]["xy"]) > 0


def estimate_image_bounds(
        image_path: Path,
        width: int,
        height: int
) -> Optional[Tuple[float, float, float, float]]:
    try:
        with get_label_path(image_path).open() as f:
            label_data = json.load(f)

        # Load all the geometries
        xy_geoms = [wkt.loads(feat["wkt"]) for feat in label_data["features"]["xy"]]
        lng_lat_geoms = [wkt.loads(feat["wkt"]) for feat in label_data["features"]["lng_lat"]]

        # Building positions
        min_x, min_y, max_x, max_y = cascaded_union(xy_geoms).bounds
        min_lon, min_lat, max_lon, max_lat = cascaded_union(lng_lat_geoms).bounds

        # Building size in pixels
        x_size = max_x - min_x
        y_size = max_y - min_y

        # Building size in degrees
        lon_size = abs(min_lon) - abs(max_lon)
        lat_size = abs(max_lat) - abs(min_lat)

        # How much of a long / lat is needed per pixel
        lons_per_pixel = (lon_size / x_size)
        lats_per_pixel = (lat_size / y_size)

        # Calculation for the bounding box of the image
        bounds_min_lon = min_lon - abs(min_x * lons_per_pixel)
        bounds_min_lat = min_lat + abs(min_y * lats_per_pixel)
        bounds_max_lon = max_lon + abs((width - max_x) * lons_per_pixel)
        bounds_max_lat = max_lat - abs((height - max_y) * lats_per_pixel)

        return bounds_min_lon, bounds_min_lat, bounds_max_lon, bounds_max_lat
    except ValueError:
        LOGGER.warn("Unable to estimate bounds for.", image_path=image_path)
        return None


def create_label_image(
        label_data: dict,
        width: int, height: int,
        bounds: Tuple[float, float, float, float]
) -> Image:
    # Create a label image based on the OSM data for the bounds.
    map_aoi = geotiler.Map(extent=bounds, size=(height, width))
    label_image = geotiler.render_map(map_aoi).convert('RGB')

    # Tint the label image with a colour based on the disaster code.
    background_image = Image.new("RGBA", (width, height))
    ImageDraw.Draw(background_image, "RGBA").polygon(
        [
            (0, 0),
            (height, 0),
            (height, width),
            (0, width)
        ],
        BACKGROUND_COLOUR_MAPPING[label_data["metadata"]["disaster"]]
    )
    label_image.paste(background_image, (0, 0), background_image)

    # Add each building to the image (with a colour corresponding to the
    # level of the damage.
    for building in label_data["features"]["xy"]:
        x, y = wkt.loads(building["wkt"]).exterior.coords.xy
        p = list(zip(x, y))
        try:
            colour = DAMAGE_COLOUR_MAPPING[building["properties"]["subtype"]]
        except KeyError:
            # In the case that the building has no 'subtype' property the
            # building is not damaged
            colour = DAMAGE_COLOUR_MAPPING["no-damage"]

        ImageDraw.Draw(label_image).polygon(p, colour)

    return label_image


def cache_label_image(image_path: Path, bounds: Tuple[float, float, float, float]):
    with get_label_path(image_path).open() as f:
        label_data = json.load(f)

    label_image = create_label_image(label_data, RAW_WIDTH, RAW_HEIGHT, bounds)
    label_image.save(get_label_image_path(image_path))


class XView2Dataset(Dataset):
    def __init__(self, directory: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._directory = directory

        LOGGER.info("Initialising samples.")
        self._samples = [
            {
                "image_path": path,
                "label_path": get_label_path(path),
                "label_image_path": get_label_image_path(path),
                "bounds": bounds
            }
            for path in list((directory / Path("images")).glob("*.png"))
            for bounds in [estimate_image_bounds(path, RAW_WIDTH, RAW_HEIGHT)]
            if get_label_image_path(path).is_file() or (has_buildings(path) and bounds is not None)
        ]
        LOGGER.info("Initialising samples completed.", sample_count=len(self._samples))

    def __getitem__(self, item):
        sample = self._samples[item]

        image_path = sample["image_path"]
        label_image_path = sample["label_image_path"]

        if not label_image_path.is_file():
            LOGGER.info("Caching image.", image_path=image_path)
            cache_label_image(image_path, sample["bounds"])
            LOGGER.info("Cached image successfully.", image_path=image_path)

        y = Image.open(image_path)
        x = Image.open(label_image_path)

        # Apply the same random crop to both the images
        i, j, h, w = transforms.RandomCrop.get_params(
            x, output_size=(INPUT_HEIGHT, INPUT_WIDTH)
        )
        x = torchvision_crop(x, i, j, h, w)
        y = torchvision_crop(y, i, j, h, w)

        return TRANSFORMS(x), TRANSFORMS(y)

    def __len__(self):
        return len(self._samples)
