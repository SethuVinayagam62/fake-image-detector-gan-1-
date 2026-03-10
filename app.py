import os
import cv2
import numpy as np
import uuid
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
IMG_SIZE = 256
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# ==========================================
# 2. FACE DETECTOR SETUP
# ==========================================
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        print("WARNING: No face detected, using full image.")
        return img_rgb

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.2 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_rgb.shape[1], x + w + pad)
    y2 = min(img_rgb.shape[0], y + h + pad)
    return img_rgb[y1:y2, x1:x2]

# ==========================================
# 3. ONNX MODEL LOADER (lazy)
# ==========================================
_session = None

def get_session():
    global _session
    if _session is None:
        print("INFO: Loading ONNX model...")
        _session = ort.InferenceSession("meso4.onnx")
        print("SUCCESS: ONNX model loaded.")
    return _session

# ==========================================
# 4. PREPROCESSING
# ==========================================
def preprocess_face(img_rgb):
    face = crop_face(img_rgb)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    face = face.astype('float32') / 255.0
    return np.expand_dims(face, axis=0)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image file.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = preprocess_face(img_rgb)

    session = get_session()
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: tensor})
    score = float(result[0][0][0])
    print(f"IMAGE raw_score={score:.4f}")
    return score

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    session = get_session()
    input_name = session.get_inputs()[0].name
    frame_scores = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 10 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess_face(img_rgb)
            result = session.run(None, {input_name: tensor})
            score = float(result[0][0][0])
            frame_scores.append(score)
            print(f"  Frame {count}: score={score:.4f}")
        count += 1

    cap.release()

    if not frame_scores:
        print("WARNING: No frames processed, defaulting to 0.5")
        return 0.5

    mean_score = float(np.mean(frame_scores))
    print(f"VIDEO mean_score={mean_score:.4f} over {len(frame_scores)} frames")
    return mean_score

# ==========================================
# 6. SCORE INTERPRETATION
# ==========================================
def interpret_score(raw_score):
    ai_percent = round((1 - raw_score) * 100, 2)

    if ai_percent < 30:
        verdict = "Likely Real"
        confidence = "High" if ai_percent < 15 else "Medium"
    elif ai_percent < 60:
        verdict = "Suspicious"
        confidence = "Low"
    else:
        verdict = "Strong AI Influence"
        confidence = "High" if ai_percent > 80 else "Medium"

    return ai_percent, verdict, confidence

# ==========================================
# 7. ROUTES
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        raw_score = predict_video(path) if is_video else predict_image(path)
        ai_percent, verdict, confidence = interpret_score(raw_score)

        return jsonify({
            "ai_influence_percent": ai_percent,
            "raw_score": round(raw_score, 4),
            "type": "video" if is_video else "image",
            "interpretation": verdict,
            "confidence": confidence
        })

    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
