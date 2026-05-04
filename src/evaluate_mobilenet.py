import os
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from load_data import build_generators_mobilenet

DATA_DIR = os.path.join('data', 'New Plant Diseases Dataset(Augmented)')
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = os.path.join('models', 'mobilenet_v2_model.h5')
REPORT_PATH = os.path.join('saved_results', 'classification_report2.txt')

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found: {MODEL_PATH}")

print(f"[INFO] Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"[INFO] Building generators from {DATA_DIR}")
train_gen, val_gen, test_gen = build_generators_mobilenet(
    DATA_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_subdir='valid',
    test_subdir='test',
)

print("[INFO] Evaluating on test generator...")
results = model.evaluate(test_gen, verbose=1)
print(f"Test loss: {results[0]:.4f}")
print(f"Test accuracy: {results[1]:.4f}")

print("[INFO] Predicting for classification report...")
preds = model.predict(test_gen)
y_true = test_gen.classes
y_pred = np.argmax(preds, axis=1)
classes = list(test_gen.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=classes)

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, 'w') as f:
    f.write(f"Test loss: {results[0]:.4f}\nTest accuracy: {results[1]:.4f}\n\n")
    f.write(report)
print(f"[INFO] Saved report to {REPORT_PATH}")
