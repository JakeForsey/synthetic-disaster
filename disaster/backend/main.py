import io
from PIL import Image
import json
import traceback

from flask import request
from flask import jsonify
from flask_cors import CORS
from flask import Flask
from flask import Response
from osmxtract import overpass
from werkzeug import FileWrapper
import torch
from torchvision import transforms
import numpy as np

from pix2pix import Generator
from data import TRANSFORMS
from data import INPUT_HEIGHT, INPUT_WIDTH
from data import create_label_image

app = Flask(__name__)
CORS(app)
generator = Generator()
try:
    # Case inside Docker container
    generator.load_state_dict(
        torch.load('checkpoints/archive/pix2pix_osm_generator_7.pth', map_location=torch.device('cpu'))
    )
except FileNotFoundError:
    # Case when invoked with python main.py
    generator.load_state_dict(
        torch.load('../../checkpoints/archive/pix2pix_osm_generator_7.pth', map_location=torch.device('cpu'))
    )

# Calling generator.eval() breaks the model.
generator.zero_grad()


class BadRequest(Exception):
    """API Request was invalid."""
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def min_max(image):
    return (image - image.min()) / (image.max() - image.min())


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.errorhandler(Exception)
def handle_exception(error):
    print(traceback.format_exc())
    response = jsonify(traceback.format_exc())
    response.status_code = 500
    return response


@app.route("/generate")
def generate():
    label_data = json.loads(request.args.get("layout"))
    min_lon = float(request.args.get("minLon"))
    max_lon = float(request.args.get("maxLon"))

    min_lat = float(request.args.get("minLat"))
    max_lat = float(request.args.get("maxLat"))

    if not label_data:
        raise BadRequest("'layout' must be provided.")
    if min_lon > max_lon:
        raise BadRequest("Min longitude is larger than max longitude.")
    if min_lat > max_lat:
        raise BadRequest("Min latitude is larger than max latitude.")

    label_image = create_label_image(
        label_data,
        INPUT_WIDTH, INPUT_HEIGHT,
        (min_lon, min_lat, max_lon, max_lat)
    )

    label_tensor = transforms.RandomCrop(
        (INPUT_HEIGHT, INPUT_WIDTH)
    )(label_image)
    label_tensor = TRANSFORMS(label_tensor)
    generated_tensor = generator(label_tensor.unsqueeze(0)).squeeze()

    generated_array = min_max(
        np.transpose(
            generated_tensor.detach().numpy(),
            (1, 2, 0)
        )
    ) * 255
    generated_image = Image.fromarray(
        generated_array.astype('uint8')
    ).convert('RGB')

    result_bytes = io.BytesIO()
    result_image = Image.new('RGB', (INPUT_WIDTH * 2, INPUT_HEIGHT))
    x_offset = 0
    for image in [label_image, generated_image]:
        result_image.paste(image, (x_offset, 0))
        x_offset += INPUT_WIDTH

    result_image.save(result_bytes, format='JPEG')
    result_bytes.seek(0)

    return Response(
        FileWrapper(result_bytes),
        mimetype="image/jpeg"
    )


@app.route("/osm")
def osm():
    min_lon = float(request.args.get("minLon"))
    max_lon = float(request.args.get("maxLon"))

    min_lat = float(request.args.get("minLat"))
    max_lat = float(request.args.get("maxLat"))

    if min_lon > max_lon:
        raise BadRequest("Min longitude is larger than max longitude.")
    if min_lat > max_lat:
        raise BadRequest("Min latitude is larger than max latitude.")

    query = overpass.ql_query(
        (min_lat, min_lon, max_lat, max_lon),
        tag="building"
    )
    response = overpass.request(query)
    feature_collection = overpass.as_geojson(response, 'polygon')

    return jsonify(feature_collection)


if __name__ == '__main__':
    app.run("0.0.0.0", port=6001, threaded=False)
