# 🛡️ Deepfake Detection System

An AI-powered Deepfake Detection web application built using Flask, TensorFlow, and OpenCV. The system analyzes uploaded images and predicts whether they are **Real** or **Deepfake (Fake)** using a trained deep learning model.

---

## 📌 Overview

Deepfake technology uses Artificial Intelligence to generate highly realistic fake images and videos. While it has many creative applications, it can also be misused for misinformation, identity fraud, and digital manipulation.

This project aims to detect manipulated facial images using a Convolutional Neural Network (CNN) trained on real and fake image datasets.

---

## 🚀 Features

- Upload images through a web interface
- AI-powered Deepfake Detection
- Real-time prediction results
- Confidence score for each prediction
- TensorFlow/Keras model integration
- Simple and user-friendly interface
- Fast image processing using OpenCV

---

## 🏗️ Project Structure

```text
DEEPFAKE/
│
├── static/
│   ├── uploads/
│   ├── css/
│   └── images/
│
├── templates/
│   └── index.html
│
├── app.py
├── deepfake_model.h5
├── Ai-Model.ipynb
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Flask

### AI / Machine Learning
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pillow

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

#### Mac/Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install flask tensorflow opencv-python numpy pillow
```

---

## ▶️ Running the Application

Start the Flask server:

```bash
python3 app.py
```

If successful, you will see:

```bash
* Running on http://127.0.0.1:5000
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🧠 How the System Works

1. User uploads an image.
2. Flask receives the image.
3. OpenCV preprocesses the image:
   - Resize image
   - Normalize pixel values
   - Convert to model input format
4. The trained CNN model analyzes facial features.
5. The model predicts whether the image is:
   - Real
   - Fake (Deepfake)
6. The result and confidence score are displayed to the user.

---

## 🤖 Model Details

- Framework: TensorFlow / Keras
- Architecture: Convolutional Neural Network (CNN)
- Input: Facial Image
- Output:
  - Real
  - Fake

Model File:

```text
deepfake_model.h5
```

Training Notebook:

```text
Ai-Model.ipynb
```

---

## 📊 Example Output

```text
Prediction: Fake
Confidence: 95.6%
```

or

```text
Prediction: Real
Confidence: 98.2%
```

---

## 🔮 Future Enhancements

- Deepfake Video Detection
- Face Region Highlighting
- Explainable AI (XAI)
- Detection History Dashboard
- User Authentication
- Cloud Deployment (AWS/Vercel/Render)
- REST API Support
- Mobile Application Integration

---

## 📋 Requirements

```text
Python 3.9+
Flask
TensorFlow
OpenCV
NumPy
Pillow
```

---

## 🎯 Applications

- Social Media Content Verification
- Fake News Detection
- Identity Fraud Prevention
- Digital Forensics
- Cybersecurity
- Media Authentication

---

## 👨‍💻 Author

### Mukul Varade

B.Tech Electronics & Telecommunication Engineering  
MIT Academy of Engineering (MITAOE), Pune

Aspiring Full-Stack & AI Engineer

---

## 📜 License

This project is intended for educational and research purposes only.

The misuse of deepfake technology for deception, fraud, harassment, or misinformation is strongly discouraged.

---
⭐ If you found this project useful, consider giving it a star on GitHub.
