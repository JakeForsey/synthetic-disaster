import random

from flask import Flask
from flask_cors import CORS
from flask import jsonify

app = Flask(__name__)
CORS(app)


@app.route("/generate")
def generate():
    return jsonify({"result": str(random.randint(0, 10))})


if __name__ == '__main__':
    app.run("0.0.0.0", port=6001)
