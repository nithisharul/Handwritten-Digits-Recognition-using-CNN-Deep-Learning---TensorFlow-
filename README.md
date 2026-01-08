Handwritten Digits Recognition using CNN , TensorFlow & Keras and using the MNIST dataset

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow and Keras** to classify handwritten digits from the **MNIST dataset**.

The model learns to recognize digits (0–9) by extracting visual features such as edges, curves, and shapes using convolution and pooling layers.



 📌 Technologies Used
 Python
 TensorFlow
 Keras (TensorFlow API)
 NumPy
 Matplotlib
 Google Colab



 📂 Dataset
 **MNIST Handwritten Digits Dataset**
 60,000 training images  
 10,000 testing images  
 Image size: **28 × 28**
 Grayscale images (1 channel)



## 🧠 Model Architecture

1. Input Layer  
    Shape: `(28, 28, 1)`

2. Convolutional Layers (3 blocks)
    Conv2D (64 filters, 3×3 kernel)
    ReLU activation
    MaxPooling (2×2)

3. Flatten Layer  
   - Converts 3D feature maps into a 1D vector

4. Fully Connected (Dense) Layers
   Dense(64) + ReLU
   Dense(32) + ReLU

5. Output Layer
   Dense(10)
   Softmax activation (multi-class classification)


📊 Data Preprocessing

- Images normalized from range **0–255 → 0–1**
- Reshaped to include channel dimension `(28, 28, 1)`
- Visualization done using Matplotlib

---

🚀 How to Run

1. Open the notebook in **Google Colab** or locally
2. Install required libraries (if needed)
3. Run all cells sequentially
4. Train the model and evaluate accuracy

---

📈 Result

The CNN Model successfully learns to classify handwritten digits with high accuracy by progressively extracting meaningful visual features.

---

📌 Key Learning Outcomes

 Understanding CNN architecture
 Difference between Conv, Pooling, Flatten, and Dense layers
 Image normalization and reshaping
 Using TensorFlow & Keras for deep learning
 Visualizing image data




-Nithish Arul**

