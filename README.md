# 🎭 Deepfake Video & Audio Detection

An end-to-end deep learning system for detecting manipulated **video and audio content** using multimodal analysis. The project analyzes facial features from video frames and acoustic patterns from audio signals to classify media as **Real** or **Deepfake**.

The system combines computer vision, deep learning, and audio signal processing to provide an additional layer of media authenticity verification.

---

## 🚀 Overview

The rapid advancement of generative AI has made it increasingly difficult to distinguish authentic media from manipulated content.

This project addresses the problem by developing separate deep learning pipelines for:

* 🎥 **Video Deepfake Detection** — analyzes facial features and temporal visual information.
* 🎙️ **Audio Deepfake Detection** — analyzes acoustic characteristics and speech patterns.
* 🔗 **Multimodal Analysis** — combines insights from different media modalities to improve authenticity assessment.

---

## ✨ Key Features

* 🎥 Video-based deepfake detection
* 👤 Face detection and extraction from video frames
* 🧠 CNN-based visual classification
* 🎙️ Audio deepfake detection
* 📊 Mel-Spectrogram based audio feature extraction
* 🔄 CNN-BiLSTM architecture for audio analysis
* 🖼️ Frame-level preprocessing and analysis
* ⚡ FastAPI backend for model inference
* 🌐 Web-based interface for uploading media
* 📈 Prediction results with confidence scores
* 🧩 Modular architecture for extending detection models

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     User Upload     │
                    │  Video / Audio File │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Media Processing  │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴────────────────┐
               │                                │
               ▼                                ▼
       ┌───────────────┐                ┌────────────────┐
       │ Video Pipeline│                │ Audio Pipeline │
       └───────┬───────┘                └───────┬────────┘
               │                                │
               ▼                                ▼
       Frame Extraction                  Audio Extraction
               │                                │
               ▼                                ▼
         Face Detection                    Preprocessing
         (MTCNN/OpenCV)                        │
               │                                ▼
               ▼                         Mel-Spectrogram
       Facial Preprocessing                     │
               │                                ▼
               ▼                          CNN-BiLSTM
          CNN / ResNet50                        │
               │                                │
               ▼                                ▼
       Video Prediction                  Audio Prediction
               │                                │
               └───────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Authenticity Result │
                    │   Real / Deepfake   │
                    └─────────────────────┘
```

---

# 🎥 Video Deepfake Detection

The video pipeline processes video content frame by frame and focuses on facial regions, where many deepfake manipulations are most visible.

### Pipeline

```text
Input Video
     ↓
Frame Extraction
     ↓
Face Detection
     ↓
Face Cropping & Preprocessing
     ↓
Deep Learning Model
     ↓
Frame-Level Predictions
     ↓
Video-Level Prediction
```

### Technologies

* Python
* OpenCV
* MTCNN
* TensorFlow / Keras
* ResNet50
* NumPy
* Computer Vision

### Processing Steps

1. Extract frames from the input video using OpenCV.
2. Detect faces using MTCNN.
3. Crop and preprocess detected facial regions.
4. Pass processed frames through the deep learning model.
5. Generate frame-level predictions.
6. Aggregate predictions to obtain the final video-level classification.

### Output

The system classifies the video as:

```text
REAL
```

or

```text
DEEPFAKE
```

along with the model's prediction confidence.

---

# 🎙️ Audio Deepfake Detection

The audio pipeline analyzes speech signals to identify patterns associated with synthetic or manipulated audio.

### Pipeline

```text
Input Audio
     ↓
Audio Extraction
     ↓
Signal Preprocessing
     ↓
Mel-Spectrogram Generation
     ↓
CNN Feature Extraction
     ↓
BiLSTM Temporal Modeling
     ↓
Classification
     ↓
Real / Deepfake
```

### Technologies

* Python
* Librosa
* TensorFlow / Keras
* CNN
* BiLSTM
* NumPy
* Signal Processing

### Processing Steps

1. Extract the audio signal from the input media.
2. Normalize and preprocess the audio.
3. Convert the waveform into Mel-Spectrogram representations.
4. Extract spatial/audio-frequency features using CNN layers.
5. Capture temporal dependencies using BiLSTM layers.
6. Classify the audio as authentic or manipulated.

---

# 🧠 Machine Learning Approach

## Video Model

The video detection pipeline uses a CNN-based architecture for extracting discriminative facial features.

**ResNet50** is used as the primary visual feature extraction/classification architecture.

The model learns visual patterns such as:

* Facial inconsistencies
* Texture abnormalities
* Blending artifacts
* Unnatural facial details
* Manipulation-related visual patterns

---

## Audio Model

The audio detection model combines:

**CNN + BiLSTM**

### CNN

Extracts meaningful patterns from Mel-Spectrogram representations.

### BiLSTM

Captures temporal dependencies in the extracted audio features by processing information in both forward and backward directions.

This combination enables the model to analyze both:

* Frequency-domain characteristics
* Temporal speech patterns

---

# 🛠️ Tech Stack

| Category             | Technologies                   |
| -------------------- | ------------------------------ |
| Programming Language | Python                         |
| Deep Learning        | TensorFlow, Keras              |
| Computer Vision      | OpenCV, MTCNN                  |
| Video Model          | ResNet50 / CNN                 |
| Audio Processing     | Librosa                        |
| Audio Model          | CNN + BiLSTM                   |
| Numerical Computing  | NumPy                          |
| Backend              | FastAPI                        |
| Frontend             | React, Tailwind CSS            |
| API                  | REST API                       |
| Development          | Jupyter Notebook, Google Colab |
| Version Control      | Git, GitHub                    |

---

# 📂 Project Structure

```text
deepfake-detection/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── models/
│
├── frontend/
│   ├── src/
│   ├── components/
│   └── pages/
│
├── video_detection/
│   ├── preprocessing/
│   ├── face_detection/
│   ├── inference/
│   └── model/
│
├── audio_detection/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── inference/
│   └── model/
│
├── notebooks/
│   ├── video_training.ipynb
│   └── audio_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact structure can be adjusted based on the final repository implementation.

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/deepfake-detection.git
cd deepfake-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

### Start the FastAPI backend

```bash
uvicorn main:app --reload
```

The API will be available locally through:

```text
http://127.0.0.1:8000
```

FastAPI automatically provides interactive API documentation through:

```text
/docs
```

### Start the frontend

```bash
npm install
npm run dev
```

The frontend can then be used to upload supported media files and view the detection results.

---

# 🔌 API Workflow

A typical inference request follows:

```text
Client
  │
  ▼
FastAPI Endpoint
  │
  ▼
File Validation
  │
  ├── Video ──► Frame Extraction ──► Face Detection ──► Model
  │
  └── Audio ──► Feature Extraction ──► CNN-BiLSTM
                                      │
                                      ▼
                              Prediction Result
                                      │
                                      ▼
                                   Client
```

Example response:

```json
{
    "prediction": "Deepfake",
    "confidence": 0.94
}
```

---

# 📊 Model Evaluation

The models can be evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

Example:

```text
                Predicted
              Real   Fake
Actual Real    TP     FN
       Fake    FP     TN
```

These metrics provide a more comprehensive evaluation than accuracy alone, particularly when the dataset contains class imbalance.

---

# 🔬 Challenges Addressed

During development, the project focused on several practical challenges associated with deepfake detection:

### 1. Unstructured Video Data

Videos contain a large number of frames, making direct processing computationally expensive.

**Approach:**
Frame sampling and targeted facial-region extraction.

### 2. Face Localization

Manipulation artifacts are often concentrated around facial regions.

**Approach:**
MTCNN and OpenCV-based face detection and preprocessing.

### 3. Audio Representation

Raw audio is not directly suitable for many deep learning architectures.

**Approach:**
Conversion of audio signals into Mel-Spectrogram representations.

### 4. Temporal Information

Audio characteristics can change over time.

**Approach:**
BiLSTM layers are used to capture temporal dependencies.

### 5. Generalization

Deepfake generation techniques vary significantly across datasets and generation methods.

**Approach:**
Use diverse preprocessing and evaluation strategies to assess model robustness.

---

# 📌 Applications

Potential applications include:

* Digital media verification
* Social media content moderation
* Journalism and fact-checking workflows
* Fraud detection
* Identity verification
* Digital forensics
* Content authenticity analysis
* Research in synthetic media detection

---

# 🔮 Future Improvements

Potential extensions include:

* Multimodal fusion of video and audio predictions
* Transformer-based architectures
* Vision Transformer (ViT) based video analysis
* Temporal video modeling
* Advanced audio embeddings
* Explainable AI for detection decisions
* Real-time video stream analysis
* Robustness testing against unseen manipulation techniques
* Model optimization for edge/mobile deployment
* Improved cross-dataset generalization

---

# ⚠️ Limitations

Deepfake detection is an evolving research problem. Detection performance can vary depending on:

* Dataset characteristics
* Video quality
* Compression
* Manipulation technique
* Audio quality
* Unseen generation methods
* Domain differences between training and real-world media

Therefore, predictions should be treated as **model-assisted authenticity assessments rather than definitive proof of manipulation**.

---

# 🎯 Learning Outcomes

Through this project, I gained hands-on experience in:

* Deep learning model development
* Computer vision
* Video preprocessing
* Face detection
* Audio signal processing
* Mel-Spectrogram feature extraction
* CNN architectures
* BiLSTM networks
* Model evaluation
* REST API development with FastAPI
* Building an end-to-end AI application
* Integrating machine learning models into production-oriented workflows

---

# 👨‍💻 Author

**Bharath G P**

Computer Science & Engineering
Bangalore, India

Interested in:

**Artificial Intelligence • Machine Learning • Generative AI • Agentic AI • Backend Engineering**

---

# 📜 License

This project is intended for educational and research purposes.

Add an appropriate open-source license such as **MIT** if you plan to distribute the source code publicly.
