# 🚦 Traffic Sign Detection using CNN

![Traffic Sign Classification](https://upload.wikimedia.org/wikipedia/commons/1/16/UK_traffic_sign_602.svg)

> A high-performance deep learning model achieving **99.83% accuracy** on the GTSRB dataset for traffic sign classification.

## 🔗 Project Link

👉 [GitHub Repository](https://github.com/Ayushsaha004/Traffic-Sign-Detector)

---

## 📌 Project Overview

This project involves the development of a **Convolutional Neural Network (CNN)** to automatically classify images of traffic signs. The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, containing 43 different traffic sign classes.

The goal is to enhance road safety by enabling real-time detection for autonomous vehicles and driver assistance systems.

📅 **Duration**: May 2025 – June 2025  
🏫 **Institution**: MCKV Institute of Engineering  
👨‍💻 **Developer**: Ayush Saha

---

## 📊 Dataset Description

- **Dataset**: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Size**: 50,000+ images
- **Classes**: 43 distinct traffic sign categories
- **Challenges**: Varying lighting conditions, angles, and backgrounds

---

## 🧹 Data Preprocessing

- Image resizing to **32x32**
- **Grayscale conversion** for simplicity
- **Histogram equalization** for contrast improvement
- **Normalization** (scaling pixel values to [0,1])
- Data split: 80% training / 20% testing

---

## 🏗️ Model Architecture

A custom CNN comprising:

- 3 Convolutional blocks (filters: 32 → 64 → 128)
- MaxPooling & Dropout layers after each block
- Dense layer with 512 neurons
- Final **Softmax output** for 43-class classification

> Built using TensorFlow and Keras

---

## 🏋️ Training Details

- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Epochs**: 80  
- **Batch Size**: 128  
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
- **Validation Split**: 20%

---

## ✅ Results

| Metric     | Score    |
|------------|----------|
| Accuracy   | **99.83%** |
| Precision  | 1.00     |
| Recall     | 1.00     |
| F1-Score   | 1.00     |

- Excellent detection of signs like **"No entry"**, **"Speed limit (20km/h)"**, etc.
- Very few errors on visually similar or rare signs.

---

## ⚠️ Limitations

- Slight misclassification in signs with low representation (e.g., “Bumpy road”).
- Performance may dip under **poor lighting or extreme angles**.
- Future improvements: Data augmentation, deeper architectures.

---

## 💻 Requirements

### Hardware
- GPU-supported system (recommended)
- Minimum 8 GB RAM
- 10 GB+ Disk Space

### Software
- **Python** ≥ 3.7
- **TensorFlow 2.x**
- NumPy, Pandas, OpenCV, Scikit-learn, Matplotlib, Seaborn, PIL
- Jupyter Notebook / Google Colab

---

## 🚀 Key Features

- ✅ **High Accuracy**: 99.83%
- 🧠 **CNN**: Custom architecture for traffic signs
- 🕵️‍♂️ **43-Class** detection capability
- 🔁 **Dropout & Regularization** for overfitting prevention
- ⚡ **Fast Inference**: Supports batch processing
- 📈 **Loss convergence**: Smooth training curves

---

## ▶️ How to Run

This project is implemented on **Kaggle**.

To run it yourself:

1. Download the entire project folder.
2. Upload it to a [Kaggle Notebook](https://www.kaggle.com/code).
3. Run the notebook.

Need help? Feel free to **reach out to me anytime**!

---

## 📚 References

- GTSRB Dataset: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- TensorFlow Documentation: https://www.tensorflow.org/
- CNN Architectures: Research on deep learning for image classification

---

## 🙋‍♂️ About Me

I'm **Ayush Saha**, a final-year student at MCKV Institute of Engineering passionate about AI, deep learning, and intelligent systems. I'm actively looking for **internship or full-time roles in Machine Learning or Software Development**.

📧 Reach me at: **ayushsahaofficial@gmail.com**  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/ayushsaha004)

---

⭐ If you like this project, don’t forget to **star** the repo!
