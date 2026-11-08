
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from predictor import predict_waste
from recommendations import get_recommendation


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "app", "static", "uploads"
)

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp"
}

MAX_FILE_SIZE = 5 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# 2. CHECK FILE FORMAT
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# 3. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 4. WASTE CLASSIFICATION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "message": "Please upload an image."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "No image selected."
        }), 400

    if not allowed_file(file.filename):

        return jsonify({
            "success": False,
            "message": "Unsupported image format."
        }), 400

    filename = secure_filename(file.filename)

    if not filename:
        return jsonify({
            "success": False,
            "message": "Invalid filename."
        }), 400

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(file_path)

        result = predict_waste(file_path)

        category = result["category"]

        recommendation = get_recommendation(category)

        return jsonify({

            "success": True,

            "prediction": result,

            "recommendation": recommendation

        })

    except (ValueError, OSError) as error:

        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

    except Exception:

        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "message": "Something went wrong during prediction."
        }), 500

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


# ==========================================
# 5. HANDLE LARGE FILES
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "message": "Image size must be below 5 MB."
    }), 413


# ==========================================
# 6. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 45)
    print("SMART WASTE CLASSIFICATION SYSTEM")
    print("Flask server starting...")
    print("=" * 45)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
