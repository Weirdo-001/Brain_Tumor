# 🧠 Brain Tumor MRI Classification using EfficientNet-B3

An AI-powered web application for classifying brain MRI scans into tumor categories using deep learning and transfer learning with EfficientNet-B3.

---

## 📌 Project Overview

This project uses a Convolutional Neural Network (CNN) based on EfficientNet-B3 to classify brain MRI images into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The trained model is deployed using Streamlit to provide a clean and interactive web interface with real-time predictions and visual analytics.

---

## 🚀 Features

- ✅ Transfer Learning using EfficientNet-B3
- ✅ Fine-tuned CNN for medical image classification
- ✅ Real-time MRI image upload and prediction
- ✅ Interactive visualizations (Gauge, Radar, Donut charts)
- ✅ Clinical information display (Symptoms, Treatment, Statistics)
- ✅ Clean and professional UI
- ✅ Cached model loading for performance optimization

---

## 🏗️ Model Architecture

**The model uses:**
- EfficientNet-B3 (Pretrained on ImageNet)
- GlobalAveragePooling2D
- BatchNormalization
- Dense (256 units, ReLU activation)
- Dropout (0.5)
- Dense (4 units, Softmax activation)

**Why EfficientNet-B3?**
- Balanced accuracy and computational efficiency
- Compound scaling (depth, width, resolution)
- Designed for 300×300 input images
- Strong feature extraction capability for medical imaging

---

## 🧠 Training Strategy

**Dataset**  
- Training & Validation split: 80–20
- Separate test dataset
- Image size: 300 × 300
- Batch size: 32

**Techniques Used**
- Data augmentation (Flip, Rotation, Zoom, Contrast)
- Transfer learning (frozen base model initially)
- Fine-tuning last 50 layers
- Adam optimizer
- Sparse categorical crossentropy
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

**Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📊 Web Application (Streamlit)

The app allows users to:
- Upload an MRI scan (JPG/PNG)
- Run AI-powered classification
- View probability distribution
- See confidence gauge
- Analyze radar chart comparison
- Explore clinical insights

**Visual Components**
- 📈 Confidence Gauge
- 🥯 Donut Probability Chart
- 📊 Radar Analysis
- 📋 Clinical Tabs (Symptoms, Treatment, Statistics)

---

## 🛠️ Tech Stack

**Backend / Model**
- Python
- TensorFlow / Keras
- EfficientNet-B3
- NumPy
- scikit-learn

**Frontend / Deployment**
- Streamlit
- Plotly
- HTML/CSS Styling
- Pillow (PIL)

---

## 📂 Project Structure

```
Brain-Tumor-Classifier/
│
├── tumor_model.keras
├── app.py
├── training_script.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

**1️⃣ Clone Repository**
```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
```

**2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

**3️⃣ Run Streamlit App**
```bash
streamlit run app.py
```

---

## 🧪 Model Performance

- **Training Accuracy:** ~87–90%
- **Validation Accuracy:** ~85–88%
- **Test Accuracy:** ~85%+
- Balanced precision and recall across tumor classes
> (Exact numbers depend on dataset version.)

---

## 🔬 How the Model Works

1. MRI image is uploaded
2. Image resized to 300×300
3. Preprocessing using EfficientNet normalization
4. CNN extracts deep features
5. Classification head predicts probabilities
6. Highest probability class selected
7. Results visualized with interactive charts

---

## 🎯 Key Learning Outcomes

- Deep understanding of CNNs
- Transfer learning and fine-tuning
- EfficientNet architecture
- Handling medical imaging datasets
- Model evaluation beyond accuracy
- Deployment using Streamlit
- UI/UX for AI applications

---

## ⚠️ Disclaimer

This application is for educational and research purposes only.  
It is not a substitute for professional medical diagnosis.  
Always consult qualified healthcare professionals for medical advice.

---

## 👩‍💻 Author

Developed as part of a deep learning and medical AI project.

---
