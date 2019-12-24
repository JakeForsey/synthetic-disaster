import io
from PIL import Image

from flask_cors import CORS
from flask import Flask
from flask import Response
from werkzeug import FileWrapper
import numpy

app = Flask(__name__)
CORS(app)


@app.route("/generate")
def generate():
    image_array = numpy.random.rand(100, 100, 3) * 255
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
