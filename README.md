# EmoScan — Real-Time Face Emotion Detection

A full-stack web app that detects faces via webcam and predicts emotions in real time
using a **deep CNN trained from scratch on FER-2013** with TensorFlow/Keras and Flask.

---

## 📁 Project Structure

```
emotion_app/
├── app.py                  # Flask backend + /predict API
├── train_model.py          # CNN training script (FER-2013)
├── requirements.txt
├── .gitignore
├── data/
│   └── fer2013.csv         # ← you place this here (from Kaggle) — not pushed to Git
├── model/
│   └── emotion_model.h5    # ← generated after training — not pushed to Git
│   └── training_curves.png # ← generated after training
├── templates/
│   └── index.html
└── static/
    ├── assets/
    │   └── logo.png
    ├── style.css
    └── script.js           # ← bug fix: emotion label now renders correctly (not mirrored)
```

---

## ⚙️ Setup Instructions

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/emoscan.git
cd emoscan/emotion_app
```

### 2 — Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow requires Python 3.8–3.11. GPU support (CUDA) is optional but speeds up training significantly.

---

## 📦 Download FER-2013 Dataset

1. Go to → https://www.kaggle.com/datasets/msambare/fer2013
2. Sign in to Kaggle and click **Download**
3. Unzip and place `fer2013.csv` inside the `data/` folder:

```
emotion_app/data/fer2013.csv
```

> This file is excluded from Git (see `.gitignore`). You must download it manually.

---

## 🧠 Train the Model

```bash
python train_model.py
```

**What this does:**
- Loads `data/fer2013.csv` (~35,000 48×48 grayscale images)
- Applies data augmentation (flip, zoom, shift, rotate)
- Trains a **deep 4-block CNN** with:
  - BatchNormalization + Dropout
  - Cosine decay LR schedule
  - Class-weight balancing (FER-2013 is imbalanced)
  - EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
- Saves best model to `model/emotion_model.h5`
- Saves training curves to `model/training_curves.png`

**Expected accuracy:** ~65–70% on the test set (FER-2013 human accuracy ~65%).

**Estimated training time:**
| Hardware | Time per epoch | Total (~30–50 epochs) |
|---|---|---|
| CPU only | ~15–20 min | 8–16 hours |
| GPU (RTX 3060) | ~45 sec | 25–40 min |

> 💡 **Tip:** Use Google Colab (free GPU) to train, download `emotion_model.h5`, then run Flask locally.

---

## 🚀 Run the Flask App

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

> If `model/emotion_model.h5` is not yet trained, the app runs in **demo mode** (random predictions so you can still test the UI pipeline).

---

## 🌐 How It Works

```
Browser                         Flask Backend
  │                                  │
  │── Click "Scan Face" ──────────►  │
  │← Webcam opens ──────────────────►│
  │                                  │
  │  [every 120ms]                   │
  │── POST /predict (base64 JPEG) ──►│
  │                        Haar Cascade face detection
  │                        Resize ROI → 48×48 grayscale
  │                        CNN forward pass
  │                        Moving-average smoothing
  │◄── JSON {faces:[{x,y,w,h,       │
  │         emotion,confidence,      │
  │         all_probs}]} ───────────►│
  │                                  │
  │  Draw bounding boxes             │
  │  Emotion label + confidence bar  │
  │  Emotion distribution panel      │
```

---

## 🎭 Emotion Classes

| # | Class | Colour |
|---|-------|--------|
| 0 | Angry | 🔴 |
| 1 | Disgust | 🟢 |
| 2 | Fear | 🟣 |
| 3 | Happy | 🟡 |
| 4 | Sad | 🔵 |
| 5 | Surprise | 🟠 |
| 6 | Neutral | 💙 |

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JS |
| Backend | Python 3.10, Flask, Flask-CORS |
| ML Model | TensorFlow 2.x / Keras |
| Face Detection | OpenCV Haar Cascade |
| Dataset | FER-2013 (Kaggle) |

---

## 📈 Model Architecture Summary

```
Input (48×48×1)
  → Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  → Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  → Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.25)
  → Conv2D(512) → BN → Conv2D(512) → BN → MaxPool → Dropout(0.25)
  → Flatten
  → Dense(512, L2) → BN → Dropout(0.5)
  → Dense(256, L2) → BN → Dropout(0.5)
  → Dense(7, softmax)
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| "Backend offline" toast | Make sure `python app.py` is running |
| Camera permission denied | Allow camera in browser settings |
| Model not loaded | Train first: `python train_model.py` |
| Low FPS | Reduce `FRAME_INTERVAL` or use GPU |
| OpenCV error | `pip install opencv-python --upgrade` |
| TF GPU not detected | Install CUDA + cuDNN matching your TF version |
| Emotion label appears mirrored | Ensure you are using the latest `static/script.js` |

---

## 🚫 What's Not Pushed to Git

The following are excluded via `.gitignore`:

```
emotion_app/data/*.csv       # FER-2013 dataset (~300MB)
emotion_app/model/*.h5       # Trained model weights (~85MB)
__pycache__/
*.pyc
.env
venv/
```

The `data/` and `model/` folders contain `.gitkeep` files so the folder structure is preserved in the repo even without the large files.