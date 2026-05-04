"""
Generate and save confusion matrix plots for both CNN and MobileNetV2 models.
Run: python src/plot_confusion_matrix.py
Requires the dataset under data/New Plant Diseases Dataset(Augmented)/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DATA_DIR = os.path.join("data", "New Plant Diseases Dataset(Augmented)")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "saved_results"


def evaluate_and_plot(model_path, model_name, preprocessing, output_dir):
    """Load model, predict on test set, save confusion matrix PNG."""

    if not os.path.exists(model_path):
        print(f"[SKIP] Model not found: {model_path}")
        return

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    model = tf.keras.models.load_model(model_path)

    # Build test generator
    test_dir = os.path.join(DATA_DIR, "test")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(DATA_DIR, "valid")

    if preprocessing == "mobilenet":
        gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        gen = ImageDataGenerator(rescale=1.0 / 255.0)

    test_gen = gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print("[INFO] Running predictions …")
    preds = model.predict(test_gen, verbose=1)
    y_true = test_gen.classes
    y_pred = np.argmax(preds, axis=1)
    class_names = list(test_gen.class_indices.keys())

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ── Full-size plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        cmap="Greens",
        xticks_rotation=90,
        values_format="d",
        colorbar=True,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=16, fontweight="bold", pad=16)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    out_path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved → {out_path}")

    # ── Normalized version ───────────────────────────────────────────
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig2, ax2 = plt.subplots(figsize=(22, 20))
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp2.plot(
        ax=ax2,
        cmap="Greens",
        xticks_rotation=90,
        values_format=".2f",
        colorbar=True,
    )
    ax2.set_title(f"{model_name} — Normalized Confusion Matrix", fontsize=16, fontweight="bold", pad=16)
    plt.tight_layout()

    out_path2 = os.path.join(output_dir, f"confusion_matrix_{safe_name}_normalized.png")
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"[INFO] Saved → {out_path2}")


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] Dataset not found at {DATA_DIR}")
        print("Download from: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        sys.exit(1)

    # CNN model
    evaluate_and_plot(
        model_path=os.path.join("models", "cnn_model.h5"),
        model_name="CNN",
        preprocessing="rescale",
        output_dir=OUTPUT_DIR,
    )

    # MobileNetV2 model
    evaluate_and_plot(
        model_path=os.path.join("models", "mobilenet_v2_model.h5"),
        model_name="MobileNetV2",
        preprocessing="mobilenet",
        output_dir=OUTPUT_DIR,
    )

    print("\n✅ Done! Check saved_results/ for confusion matrix PNGs.")


if __name__ == "__main__":
    main()
