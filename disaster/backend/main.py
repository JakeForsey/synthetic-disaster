import io
from PIL import Image

from flask_cors import CORS
from flask import Flask
from flask import Response
from werkzeug import FileWrapper
import torch

from cgan import Generator
from cgan import generate_digits


app = Flask(__name__)
CORS(app)
digit_generator = Generator()
digit_generator.load_state_dict(
    torch.load('models/digits/generator_state.pt', map_location=torch.device('cpu'))
)
digit_generator.eval()


@app.route("/generate")
def generate():
    fake_images, fake_labels = generate_digits(digit_generator, 1, )

    image_array = fake_images[0].detach().numpy() * 255
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
