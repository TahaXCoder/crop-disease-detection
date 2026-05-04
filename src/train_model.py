import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
import traceback

def ensure_dataset_structure(data_dir, train_subdir="train", val_subdir="valid", test_subdir="test"):
    train_dir = os.path.join(data_dir, train_subdir)
    val_dir = os.path.join(data_dir, val_subdir)
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory missing: {train_dir}")
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if len(train_classes) == 0:
        raise FileNotFoundError(f"No class subfolders found in train directory: {train_dir}")
    if not os.path.isdir(val_dir):
        print(f"[INFO] Validation directory not found at {val_dir}; using validation_split from train.")
    else:
        val_classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
        if len(val_classes) == 0:
            raise FileNotFoundError(f"No class subfolders found in validation directory: {val_dir}")
    test_dir = os.path.join(data_dir, test_subdir)
    if not os.path.isdir(test_dir):
        print(f"[INFO] Test directory not found at {test_dir}; will reuse validation generator for testing.")

def build_generators(data_dir, img_size, batch_size, train_subdir="train", val_subdir="valid", test_subdir="test"):
    # Ensure target size is a tuple (h, w) even if an int is passed
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    train_dir = os.path.join(data_dir, train_subdir)
    val_dir = os.path.join(data_dir, val_subdir)
    test_dir = os.path.join(data_dir, test_subdir)

    # Train / Val
    if os.path.isdir(val_dir):
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
        # Use a split from train if no separate val folder
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

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

    # Test: use test_dir if it exists, otherwise reuse val
    if os.path.isdir(test_dir):
        test_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
            directory=test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
    else:
        test_generator = val_generator

    return train_generator, val_generator, test_generator

def create_model(input_shape, num_classes):
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False  # freeze backbone for a quick transfer-learning run

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def main():
    # Default to the bundled PlantVillage-style dataset. Adjust if you move data.
    DATA_DIR = os.path.join("data", "New Plant Diseases Dataset(Augmented)")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 5

    ensure_dataset_structure(DATA_DIR, val_subdir="valid", test_subdir="test")
    print(f"[INFO] Building generators from {DATA_DIR} (img_size={IMG_SIZE}, batch_size={BATCH_SIZE})")
    train_gen, val_gen, test_gen = build_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE, val_subdir="valid", test_subdir="test")
    num_classes = train_gen.num_classes
    print(f"[INFO] Detected classes: {list(train_gen.class_indices.keys())}")

    model = create_model(input_shape=IMG_SIZE + (3,), num_classes=num_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

    print("[INFO] Evaluating on test/validation data...")
    eval_results = model.evaluate(test_gen, verbose=1)

    # save model
    model_out = os.path.join(os.path.dirname(__file__), "..", "models", "mobilenet_v2_model.h5")
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)
    print(f"[INFO] Model saved to: {model_out}")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "saved_results")
    os.makedirs(results_dir, exist_ok=True)
    preds = model.predict(test_gen)
    y_true = test_gen.classes
    y_pred = np.argmax(preds, axis=1)
    target_names = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=target_names)
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test loss: {eval_results[0]:.4f}\nTest accuracy: {eval_results[1]:.4f}\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to: {report_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)