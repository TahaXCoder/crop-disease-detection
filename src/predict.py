import os
from pathlib import Path
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_model.h5"
CLASS_NAMES = None  # will load from generator if available

def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path: Path, target_size=(224, 224)):
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(model, img_path: Path, class_names=None):
    batch = preprocess_image(img_path)
    preds = model.predict(batch)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds, axis=1)[0])
    label = class_names[idx] if class_names else str(idx)
    return label, prob

def main():
    model = load_model()
    # If you have class names saved elsewhere, load them here. Otherwise, indices are returned.
    sample_dir = Path("data")
    sample = None
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        found = list(sample_dir.rglob(ext))
        if found:
            sample = found[0]
            break
    if not sample:
        raise SystemExit("No sample image found under data/. Place an image to test.")

    label, prob = predict_image(model, sample, class_names=CLASS_NAMES)
    print(f"Sample: {sample}")
    print(f"Prediction: {label} (p={prob:.4f})")

if __name__ == "__main__":
    main()
