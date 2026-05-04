import os
import sys
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = os.path.join("data", "New Plant Diseases Dataset(Augmented)")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = os.path.join("models", "cnn_model.h5")
REPORT_PATH = os.path.join("saved_results", "classification_report.txt")


def build_simple_generators(data_dir, img_size, batch_size, val_subdir="valid", test_subdir="test"):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, val_subdir)
    test_dir = os.path.join(data_dir, test_subdir)

    rescale = 1.0 / 255.0
    # If val exists, use separate val; otherwise split from train
    if os.path.isdir(val_dir):
        train_gen = ImageDataGenerator(rescale=rescale)
        val_gen = ImageDataGenerator(rescale=rescale)
        train_generator = train_gen.flow_from_directory(
            directory=train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
        )
        val_generator = val_gen.flow_from_directory(
            directory=val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
    else:
        train_gen = ImageDataGenerator(rescale=rescale, validation_split=0.2)
        train_generator = train_gen.flow_from_directory(
            directory=train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )
        val_generator = train_gen.flow_from_directory(
            directory=train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

    if os.path.isdir(test_dir):
        test_gen = ImageDataGenerator(rescale=rescale)
        test_generator = test_gen.flow_from_directory(
            directory=test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
    else:
        test_generator = val_generator

    return train_generator, val_generator, test_generator


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"[INFO] Building generators from {DATA_DIR}")
    _, val_gen, test_gen = build_simple_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE, val_subdir="valid", test_subdir="test")

    print("[INFO] Evaluating on test/validation data...")
    results = model.evaluate(test_gen, verbose=1)
    loss, acc = results[0], results[1]
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    print("[INFO] Generating classification report...")
    preds = model.predict(test_gen)
    y_true = test_gen.classes
    y_pred = np.argmax(preds, axis=1)
    target_names = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=target_names)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Test loss: {loss:.4f}\nTest accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
