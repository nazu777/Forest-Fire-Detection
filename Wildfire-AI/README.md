# Wildfire AI

A lightweight deep learning model for wildfire detection using **Sentinel-2 multispectral satellite imagery**. The model is based on a custom 10-layer CNN inspired by MobileNetV2 and optimized for edge deployment.

---

## Features

- Wildfire vs Non-Wildfire Classification
- Supports 4-channel Sentinel-2 images (RGB + SWIR)
- ImageNet weight initialization with SWIR weight transfer
- Lightweight architecture suitable for embedded deployment
- TensorFlow implementation
- End-to-end training and inference pipeline

---

## Project Structure

```
Wildfire-AI
│
├── data/
│   ├── sample/
│   └── dataset_links.md
│
├── models/
│   └── 10_Layers.h5
│
├── results/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The complete dataset is too large to upload to GitHub.

Download using the links below.

### Forest Fire

Part 1

https://drive.google.com/drive/folders/10iWP3iZXk8d2f83L0P1tQQytKgpjamCX

Part 2

https://drive.google.com/drive/folders/10pRszrWARyerl2Q2hY4-PLHKfji-L4Cx

Part 3

https://drive.google.com/drive/folders/1FajAGdKD_xTWl2Hk-Ry8T4gO2ICqsK24

### Non Forest Fire

Part 1

https://drive.google.com/drive/folders/1tFs7ikeQi7Qh7-v5J8x03hqK5rgRfCcO

Part 2

https://drive.google.com/drive/folders/1Kmg4dBaQDuI-T3Dy6vX1jHuQRDwCnpAV

### Volcano

https://drive.google.com/drive/folders/1Zc1kQM7-JpLWOT3YLNgBVj-umBcddyIW

### Industry Clear

http://drive.google.com/drive/folders/1gamsnDOqIMZrlmqdKfce2uttlQ1CN1LP

### Industry Fire

https://drive.google.com/drive/folders/1tokYo4nWES1_chPfoputhqy3VKN4QQq1

### Urban Cities

https://drive.google.com/drive/folders/1O75CjzLCrGVHohp765tlnC47_RcsaBXN

---

## Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/Wildfire-AI.git

cd Wildfire-AI
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Dataset Layout

```
data/
└── dataset_all/
    ├── forest_fire/
    └── non_forest_fire/
```

---

## Training

Run

```bash
python src/train.py
```

The trained model will be saved inside

```
models/
```

---

## Prediction

Predict a single image

```bash
python src/predict.py --image data/sample/fire/sample.tif
```

Example Output

```
Prediction Score : 0.9821

Prediction : Forest Fire
```

---

## Model Architecture

Input

↓

Conv2D

↓

BatchNorm

↓

ReLU

↓

DepthwiseConv2D

↓

BatchNorm

↓

ReLU

↓

Pointwise Conv

↓

Global Average Pooling

↓

Dense(32)

↓

Dense(1)

---

## Technologies

- Python
- TensorFlow
- NumPy
- Scikit-Learn
- MobileNetV2
- Sentinel-2 Imagery

---

## Future Improvements

- Multi-class wildfire severity detection
- Real-time satellite stream inference
- Explainable AI using Grad-CAM
- FPGA deployment using hls4ml
- Integration with GIS platforms

---

## Author

Shrihari Viswanathan

GitHub

https://github.com/ShrihariViswanathan

LinkedIn

https://www.linkedin.com/in/shrihari-viswanathan-3602a330a/