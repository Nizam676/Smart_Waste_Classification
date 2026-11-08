
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# MODEL 1: CUSTOM CNN
# ==========================================

def build_custom_cnn(num_classes=6):

    model = models.Sequential([

        layers.Input(shape=(224, 224, 3)),

        # Data normalization
        layers.Rescaling(1.0 / 255),

        # Convolution Block 1
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 4
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Classification layers
        layers.Flatten(),

        layers.Dense(256, activation="relu"),

        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")

    ], name="Custom_CNN")

    return model


# ==========================================
# MODEL 2: MOBILENETV2
# ==========================================

def build_mobilenetv2(num_classes=6):

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # MobileNetV2 preprocessing
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    )(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_Waste_Classifier"
    )

    return model


# ==========================================
# MODEL COMPILATION
# ==========================================

def compile_model(model, learning_rate=0.0001):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model
