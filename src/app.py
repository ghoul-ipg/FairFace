import traceback

from flask import Flask, request, jsonify

from fairface_service import FairFace, FairFaceError

app = Flask(__name__)
fairface = FairFace()


@app.route("/")
def check():
    return "Work on"


@app.route("/fairface", methods=["POST"])
def process_image():
    image = request.files.get('image')
    try:
        return jsonify(fairface.process_image(image.read()))
    except FairFaceError as e:
        print(traceback.format_exc())
        return jsonify({"error_message": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0')
