# 🧠 YOLOv1: Object Detection from Scratch

An end-to-end implementation of **YOLOv1 (You Only Look Once)** object detection model, built from scratch using **PyTorch** and trained on the **Pascal VOC dataset**. This project demonstrates YOLO's ability to predict bounding boxes and class probabilities directly from images in a single forward pass.

## 🚀 Project Description
YOLO (You Only Look Once) is a real-time object detection algorithm known for its speed and accuracy. This implementation follows the YOLOv1 architecture, showcasing the methodology of dividing images into grid cells, predicting bounding boxes, and optimizing detection accuracy.

## 🛠️ Key Features
- Full **YOLOv1 model architecture** implemented from scratch.
- Dataset preprocessing and augmentation.
- Training pipeline with **loss functions for classification, localization, and confidence scores**.
- Evaluation metrics for object detection performance.
- Visualization of bounding boxes on detected objects.

## 📚 Libraries and Dependencies
- **PyTorch**: Model building and training
- **Torchvision**: Dataset transformations
- **Matplotlib**: Visualization
- **Pandas**: Data manipulation
- **Pillow (PIL)**: Image processing
- **TQDM**: Progress bar for training

## 📦 Dataset
- **Pascal VOC Dataset**
- Downloaded via Kaggle Datasets

## ⚙️ Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolov1-from-scratch.git
cd yolov1-from-scratch
```
2. Install the dependencies:
```bash
pip install torch torchvision matplotlib pandas pillow tqdm
```
3. Download the Pascal VOC dataset:
```bash
kaggle datasets download -d aladdinpersson/pascalvoc-yolo
unzip pascalvoc-yolo.zip
```

## 📊 Model Architecture
- **Input Image:** Divided into grid cells.
- **Feature Extraction:** Through Convolutional Neural Networks (CNNs).
- **Bounding Box Prediction:** Each grid predicts multiple bounding boxes.
- **Loss Function:** Balances classification, localization, and confidence loss.

## 📈 Results
- Visualization of predicted bounding boxes.
- Evaluation metrics: mAP, IoU, etc.

## 🤝 Contributions
Feel free to fork the repository, raise issues, or suggest improvements.

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
For feedback or queries, reach out at **your.email@example.com**.

---
**Real-time object detection, simplified. 🚀**
