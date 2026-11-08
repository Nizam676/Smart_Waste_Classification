
import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from model import (
    build_custom_cnn,
    build_mobilenetv2,
    compile_model
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = "data/dataset-resized"
MODEL_DIR = "models"
RESULTS_DIR = "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

print("\n" + "=" * 55)
print("SMART WASTE CLASSIFICATION SYSTEM")
print("TRAINING AND MODEL COMPARISON")
print("=" * 55)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading TrashNet dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nWaste categories:", class_names)
print("Number of classes:", num_classes)

if num_classes != 6:
    raise ValueError(
        f"Expected 6 categories, found {num_classes}"
    )

# Save category names
with open(
    os.path.join(MODEL_DIR, "class_names.json"),
    "w"
) as file:
    json.dump(class_names, file, indent=4)

# Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 3. MODEL TRAINING FUNCTION
# ==========================================

def train_and_evaluate(model, model_name):

    print("\n" + "=" * 55)
    print(f"TRAINING: {model_name}")
    print("=" * 55)

    compile_model(model)

    model.summary()

    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Load best checkpoint
    best_model = tf.keras.models.load_model(
        model_path
    )

    # Evaluate
    loss, accuracy = best_model.evaluate(
        val_ds,
        verbose=1
    )

    # Generate predictions
    y_true = []
    y_pred = []

    for images, labels in val_ds:

        predictions = best_model.predict(
            images,
            verbose=0
        )

        y_true.extend(labels.numpy())
        y_pred.extend(
            np.argmax(predictions, axis=1)
        )

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    print(f"\n{model_name} RESULTS")
    print("-" * 45)

    print(
        "Validation Accuracy:",
        round(accuracy * 100, 2), "%"
    )

    print(
        "Validation Loss:",
        round(loss, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(matrix)

    # Save evaluation results
    results = {
        "model": model_name,
        "validation_accuracy": float(accuracy),
        "validation_loss": float(loss),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "training_history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        }
    }

    results_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_results.json"
    )

    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    print(f"\nModel saved: {model_path}")
    print(f"Results saved: {results_path}")

    return accuracy


# ==========================================
# 4. TRAIN CUSTOM CNN
# ==========================================

custom_cnn = build_custom_cnn(
    num_classes=num_classes
)

custom_accuracy = train_and_evaluate(
    custom_cnn,
    "custom_cnn"
)

del custom_cnn
tf.keras.backend.clear_session()


# ==========================================
# 5. TRAIN MOBILENETV2
# ==========================================

mobilenet = build_mobilenetv2(
    num_classes=num_classes
)

mobilenet_accuracy = train_and_evaluate(
    mobilenet,
    "mobilenetv2"
)

del mobilenet
tf.keras.backend.clear_session()


# ==========================================
# 6. FINAL MODEL COMPARISON
# ==========================================

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

print(
    f"Custom CNN Accuracy: {custom_accuracy * 100:.2f}%"
)

print(
    f"MobileNetV2 Accuracy: {mobilenet_accuracy * 100:.2f}%"
)

if mobilenet_accuracy > custom_accuracy:
    best_model_name = "mobilenetv2"
else:
    best_model_name = "custom_cnn"

print("\nBetter validation accuracy:", best_model_name)

comparison = {
    "custom_cnn_accuracy": float(custom_accuracy),
    "mobilenetv2_accuracy": float(mobilenet_accuracy),
    "higher_validation_accuracy": best_model_name
}

with open(
    os.path.join(RESULTS_DIR, "model_comparison.json"),
    "w"
) as file:
    json.dump(comparison, file, indent=4)

print("\nTraining and comparison completed!")
print("=" * 55)
