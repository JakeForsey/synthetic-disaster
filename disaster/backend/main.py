import io
from PIL import Image
import json

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


def min_max(image):
    return (image - image.min()) / (image.max() - image.min())


@app.route("/generate")
def generate():
    label_data = json.loads(request.args.get("layout"))
    min_lon = float(request.args.get("minLon"))
    max_lon = float(request.args.get("maxLon"))

    min_lat = float(request.args.get("minLat"))
    max_lat = float(request.args.get("maxLat"))

    assert min_lon < max_lon, "Min longitude is smaller than max longitude"
    assert min_lat < max_lat, "Min latitude is smaller than max latitude"

    label_image = create_label_image(label_data, INPUT_WIDTH, INPUT_HEIGHT, (min_lon, min_lat, max_lon, max_lat))

    label_tensor = transforms.RandomCrop(
        (INPUT_HEIGHT, INPUT_WIDTH)
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


@app.route("/osm")
def osm():
    min_lon = request.args.get("minLon")
    max_lon = request.args.get("maxLon")

    min_lat = request.args.get("minLat")
    max_lat = request.args.get("maxLat")

    assert min_lon is not None, "Min longitude is missing."
    assert max_lon is not None, "Max longitude is missing."
    assert min_lat is not None, "Min latitude is missing."
    assert max_lat is not None, "Max latitude is missing."

    assert min_lon > max_lon, "Min longitude is larger than max longitude"
    assert min_lat < max_lat, "Min latitude is larger than max latitude"

    query = overpass.ql_query(
        (min_lat, min_lon, max_lat, max_lon),
        tag="building"
    )
    response = overpass.request(query)
    feature_collection = overpass.as_geojson(response, 'polygon')

    # Write as GeoJSON
    with open('/tmp/features.geojson', 'w') as f:
        json.dump(feature_collection, f)

    return jsonify(feature_collection)


if __name__ == '__main__':
    app.run("0.0.0.0", port=6001, threaded=False)
