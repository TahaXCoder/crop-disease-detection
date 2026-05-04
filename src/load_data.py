"""Utilities for loading PlantVillage images."""
from pathlib import Path
from typing import Tuple
import os

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_image_generators(
    data_root: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.15,
):
    """Create Keras ImageDataGenerators for train, val, and test data."""
    data_root = Path(data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected train and test directories inside data root")

    rescale = 1.0 / 255.0
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=rescale,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="training",
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        subset="validation",
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=rescale)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen

def build_generators(data_dir: str, img_size: int = 224, batch_size: int = 32, val_subdir: str = "valid", test_subdir: str = "test"):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, val_subdir)
    test_dir = os.path.join(data_dir, test_subdir)

    rescale = 1.0 / 255.0
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=rescale,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=rescale)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    if os.path.isdir(test_dir):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=rescale)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
    else:
        test_gen = val_gen

    return train_gen, val_gen, test_gen


def build_generators_mobilenet(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_subdir: str = "valid",
    test_subdir: str = "test",
):
    """Generators with MobileNetV2 preprocessing (ImageNet normalization)."""

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, val_subdir)
    test_dir = os.path.join(data_dir, test_subdir)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    if os.path.isdir(test_dir):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
    else:
        test_gen = val_gen

    return train_gen, val_gen, test_gen
