import io
from PIL import Image
import json
from flask import request

from flask_cors import CORS
from flask import Flask
from flask import Response
from werkzeug import FileWrapper
import torch
from torchvision import transforms
import numpy as np

from pix2pix import Generator
from bin.train_pix2pix import XView2Dataset
from bin.train_pix2pix import HEIGHT, WIDTH

app = Flask(__name__)
CORS(app)
generator = Generator()
generator.load_state_dict(
    torch.load('checkpoints/archive/pix2pix_generator_115.pth', map_location=torch.device('cpu'))
)
# Calling generator.eval() breaks the model.
generator.zero_grad()

TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def min_max(image):
    return (image - image.min()) / (image.max() - image.min())


@app.route("/generate")
def generate():
    label_data = json.loads(request.args.get("layout"))
    label_image = XView2Dataset.create_label_image(label_data, WIDTH, HEIGHT)

    label_tensor = transforms.RandomCrop(
        (HEIGHT, WIDTH)
    )(label_image)
    label_tensor = TRANSFORMS(label_tensor)
    fake_image = generator(label_tensor.unsqueeze(0)).squeeze()

    image_array = min_max(
        np.transpose(
            fake_image.detach().numpy(),
            (1, 2, 0)
        )
    ) * 255

    image_bytes = io.BytesIO()
    Image.fromarray(
        image_array.astype('uint8')
    ).convert(
        'RGB'
    ).save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    return Response(
        FileWrapper(image_bytes),
        mimetype="image/jpeg"
    )


if __name__ == '__main__':
    app.run("0.0.0.0", port=6001)
