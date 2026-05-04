---
title: Crop Disease Detection
emoji: 🌿
colorFrom: green
colorTo: green
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---

# 🌿 Crop Disease Detection

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

> AI-powered crop disease detection using deep learning. Upload a leaf image and get an instant diagnosis across **38 disease classes** and **14 plant species**.

## 🚀 Live Demo

**👉 [Try it now on Streamlit Cloud](https://crop-disease-detection.streamlit.app)**

---

## 📸 Screenshots

| Upload & Predict | Top-K Results |
|:-:|:-:|
| Upload a leaf photo and get instant diagnosis | See confidence scores for top predictions |

---

## 🏗️ Project Structure

```
crop-disease-detection/
├── streamlit_app.py              # 🌐 Streamlit web app (main entry)
├── app/
│   └── app.py                    # Flask app (legacy)
├── src/
│   ├── load_data.py              # Data generators (rescale & MobileNet)
│   ├── train_model.py            # MobileNetV2 training script
│   ├── evaluate_cnn.py           # CNN evaluation + classification report
│   ├── evaluate_mobilenet.py     # MobileNetV2 evaluation
│   ├── predict.py                # Single-image prediction utility
│   ├── plot_confusion_matrix.py  # 📊 Confusion matrix generator
│   └── utils.py                  # Plotting & reporting helpers
├── models/
│   ├── cnn_model.h5              # Custom CNN (231 MB)
│   └── mobilenet_v2_model.h5     # MobileNetV2 (10 MB) ← used by Streamlit
├── saved_results/
│   ├── classification_report.txt       # CNN per-class metrics
│   ├── classification_report2.txt      # MobileNetV2 per-class metrics
│   ├── accuracy_plot.png               # Training curves
│   ├── loss_plot.png                   # Loss curves
│   ├── confusion_matrix_cnn.png        # 📊 CNN confusion matrix
│   └── confusion_matrix_mobilenetv2.png# 📊 MobileNetV2 confusion matrix
├── data/
│   └── New Plant Diseases Dataset(Augmented)/
│       ├── train/
│       ├── valid/
│       └── test/
├── notebooks/
│   └── data_preprocessing.ipynb
├── .streamlit/
│   └── config.toml               # Streamlit theme config
├── requirements.txt
└── README.md
```

---

## 🤖 Trained Models

| Model | Architecture | Test Accuracy | Test Loss | Size |
|-------|-------------|:------------:|:---------:|:----:|
| **CNN** | 5×Conv/MaxPool → Dense 512 + Dropout 0.5 → Softmax | **95.01%** | 0.1533 | 231 MB |
| **MobileNetV2** | ImageNet backbone (frozen) + GAP + Dropout 0.3 → Softmax | **95.31%** | 0.1395 | 10 MB |

The Streamlit app uses **MobileNetV2** for its compact size and slightly higher accuracy.

---

## 📊 Confusion Matrix

Generate confusion matrix plots for both models:

```bash
python src/plot_confusion_matrix.py
```

This creates four files in `saved_results/`:
- `confusion_matrix_cnn.png` — raw counts
- `confusion_matrix_cnn_normalized.png` — normalized (%)
- `confusion_matrix_mobilenetv2.png` — raw counts
- `confusion_matrix_mobilenetv2_normalized.png` — normalized (%)

---

## 📦 Dataset Setup

This repository does **not** include the dataset due to its large size.

### Download the Dataset
1. Go to [Kaggle — New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. Download and unzip it
3. Place it in the following structure:

```text
crop-disease-detection/
└── data/
    └── New Plant Diseases Dataset(Augmented)/
        ├── train/
        ├── valid/
        └── test/
```

### Download Trained Models
The trained model weights are not included in this repo. Either train from scratch:

```bash
python src/train_model.py
```

Or contact me to get the pretrained `.h5` files.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/TahaXCoder/crop-disease-detection.git
cd crop-disease-detection
python -m venv .venv
.venv\Scripts\Activate     # Windows
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. Upload any leaf photo to get a diagnosis.

### 3. Evaluate Models

```bash
# CNN evaluation
python src/evaluate_cnn.py

# MobileNetV2 evaluation
python src/evaluate_mobilenet.py

# Generate confusion matrices
python src/plot_confusion_matrix.py
```

---

## ☁️ Deploy on Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**
2. Name your space (e.g. `crop-disease-detection`)
3. Choose **Streamlit** as the Space SDK
4. Leave Space hardware as **Blank/Free** and click **Create Space**
5. Once created, push this repository directly to the Hugging Face remote, or link it via GitHub Actions!
   
> **Note:** The `README.md` already contains the required YAML header so Hugging Face will automatically detect `streamlit_app.py` as your main app file! The MobileNetV2 model (10 MB) is also perfectly sized for the free tier.

---

## 🧬 Supported Classes (38)

<details>
<summary>Click to expand full class list</summary>

| # | Class |
|---|-------|
| 1 | Apple — Apple scab |
| 2 | Apple — Black rot |
| 3 | Apple — Cedar apple rust |
| 4 | Apple — Healthy |
| 5 | Blueberry — Healthy |
| 6 | Cherry — Powdery mildew |
| 7 | Cherry — Healthy |
| 8 | Corn — Cercospora leaf spot / Gray leaf spot |
| 9 | Corn — Common rust |
| 10 | Corn — Northern Leaf Blight |
| 11 | Corn — Healthy |
| 12 | Grape — Black rot |
| 13 | Grape — Esca (Black Measles) |
| 14 | Grape — Leaf blight (Isariopsis) |
| 15 | Grape — Healthy |
| 16 | Orange — Haunglongbing (Citrus greening) |
| 17 | Peach — Bacterial spot |
| 18 | Peach — Healthy |
| 19 | Pepper — Bacterial spot |
| 20 | Pepper — Healthy |
| 21 | Potato — Early blight |
| 22 | Potato — Late blight |
| 23 | Potato — Healthy |
| 24 | Raspberry — Healthy |
| 25 | Soybean — Healthy |
| 26 | Squash — Powdery mildew |
| 27 | Strawberry — Leaf scorch |
| 28 | Strawberry — Healthy |
| 29 | Tomato — Bacterial spot |
| 30 | Tomato — Early blight |
| 31 | Tomato — Late blight |
| 32 | Tomato — Leaf Mold |
| 33 | Tomato — Septoria leaf spot |
| 34 | Tomato — Spider mites |
| 35 | Tomato — Target Spot |
| 36 | Tomato — Yellow Leaf Curl Virus |
| 37 | Tomato — Mosaic virus |
| 38 | Tomato — Healthy |

</details>

---

## 📄 License

This project is for educational purposes. The PlantVillage dataset is publicly available on Kaggle.

---

<p align="center">
  Built with ❤️ using TensorFlow, Streamlit &amp; PlantVillage Dataset
</p>
