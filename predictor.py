
import os
import json
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenetv2.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

IMAGE_SIZE = (224, 224)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

print("Loading MobileNetV2 model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(
        f"Class names not found: {CLASS_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("MobileNetV2 loaded successfully!")
print("Categories:", class_names)


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path):

    try:
        image = Image.open(image_path).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from error


# ==========================================
# 4. WASTE PREDICTION
# ==========================================

def predict_waste(image_path):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            "Image file does not exist."
        )

    image_array = preprocess_image(image_path)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_category = class_names[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )

    # Confidence of all categories
    all_predictions = {
        class_names[index]: round(
            float(predictions[index] * 100), 2
        )
        for index in range(len(class_names))
    }

    return {
        "category": predicted_category,
        "confidence": round(confidence, 2),
        "all_predictions": all_predictions
    }


# ==========================================
# 5. TESTING
# ==========================================

if __name__ == "__main__":

    print("\nPrediction module is ready.")

    print("Available categories:")

    for category in class_names:
        print("-", category)
