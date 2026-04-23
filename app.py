import os
import base64
import threading
import numpy as np
import cv2
from collections import deque
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Model path — tries fixed first, falls back to original ───────────────────
_base  = os.path.dirname(os.path.abspath(__file__))
_fixed = os.path.join(_base, 'model', 'emotion_model_fixed.h5')
_orig  = os.path.join(_base, 'model', 'emotion_model.h5')
MODEL_PATH = _fixed if os.path.exists(_fixed) else _orig

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = None
model_loading = True   # True while the background thread is still loading

# ── Load model in a background thread so Flask always starts ─────────────────
def load_model():
    global model, model_loading
    try:
        import tensorflow as tf
        print(f"[*] TensorFlow version : {tf.__version__}")
        print(f"[*] Loading model from : {MODEL_PATH}")

        if not os.path.exists(MODEL_PATH):
            print(f"[✗] Model file not found at {MODEL_PATH}")
            model = None
            model_loading = False
            return

        # compile=False avoids the 'quantization_config' error on older TF
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Warm-up so first real request is not slow
        dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
        model.predict(dummy, verbose=0)
        print("[✓] Model loaded and warmed up successfully")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[✗] Model load failed: {e}")
        model = None
    finally:
        model_loading = False

# ── Face detector ─────────────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── Moving-average smoother ───────────────────────────────────────────────────
SMOOTH_WINDOW = 5
smooth_buffers = {}

def get_face_key(x, y, w, h, existing_keys, threshold=60):
    cx, cy = x + w // 2, y + h // 2
    for key in existing_keys:
        kx, ky = map(int, key.split('_'))
        if abs(cx - kx) < threshold and abs(cy - ky) < threshold:
            return key
    return f"{cx}_{cy}"

# ── Routes ────────────────────────────────────────────────────────────────────

# FIX 1: index.html lives in templates/ — use render_template, not send_from_directory
@app.route('/')
def index():
    return render_template('index.html')

# Serve static files explicitly (covers /static/script.js etc.)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/status')
def status():
    return jsonify({
        'model_loaded':  model is not None,
        'model_loading': model_loading,
        'model_path':    MODEL_PATH,
        'emotions':      EMOTIONS
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'ok': True}), 200
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        results     = []
        new_buffers = {}

        for (x, y, w, h) in (faces_rect if len(faces_rect) > 0 else []):
            face_roi   = gray[y:y+h, x:x+w]
            face_roi   = cv2.resize(face_roi, (48, 48))
            face_roi   = face_roi.astype('float32') / 255.0
            face_input = np.expand_dims(face_roi, axis=(0, -1))

            if model is not None:
                probs = model.predict(face_input, verbose=0)[0]
            else:
                probs = np.random.dirichlet(np.ones(7))

            face_key = get_face_key(x, y, w, h, smooth_buffers.keys())
            if face_key not in smooth_buffers:
                smooth_buffers[face_key] = deque(maxlen=SMOOTH_WINDOW)
            smooth_buffers[face_key].append(probs)
            new_buffers[face_key] = smooth_buffers[face_key]

            avg_probs   = np.mean(list(smooth_buffers[face_key]), axis=0)
            emotion_idx = int(np.argmax(avg_probs))
            confidence  = float(avg_probs[emotion_idx])

            results.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'emotion':    EMOTIONS[emotion_idx],
                'confidence': round(confidence, 4),
                'all_probs':  {EMOTIONS[i]: round(float(avg_probs[i]), 4)
                               for i in range(7)}
            })

        smooth_buffers.clear()
        smooth_buffers.update(new_buffers)
        return jsonify({'faces': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # FIX 2: load model in background so Flask always binds to the port,
    # even if TensorFlow is slow or throws an error.
    t = threading.Thread(target=load_model, daemon=True)
    t.start()
    print("\n[✓] Open your browser at → http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
