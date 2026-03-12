import os
import cv2
import json
import uuid
import numpy as np
import onnxruntime as ort
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
IMG_SIZE        = 256
UPLOAD_FOLDER   = "uploads"
HISTORY_FILE    = "scan_history.json"
COMPLAINTS_FILE = "complaints.json"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "checkai_secret_2025")
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB

# ==========================================
# 2. FACE DETECTOR SETUP
# ==========================================
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face(img_rgb):
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        print("WARNING: No face detected, using full image.")
        return img_rgb
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.2 * max(w, h))
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(img_rgb.shape[1], x + w + pad)
    y2  = min(img_rgb.shape[0], y + h + pad)
    return img_rgb[y1:y2, x1:x2]

# ==========================================
# 3. ONNX MODEL LOADER (lazy singleton)
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
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image file.")
    img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor     = preprocess_face(img_rgb)
    session    = get_session()
    input_name = session.get_inputs()[0].name
    result     = session.run(None, {input_name: tensor})
    score      = float(result[0][0][0])
    print(f"IMAGE raw_score={score:.4f}")
    return score


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    session      = get_session()
    input_name   = session.get_inputs()[0].name
    frame_scores = []
    count        = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 10 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor  = preprocess_face(img_rgb)
            result  = session.run(None, {input_name: tensor})
            score   = float(result[0][0][0])
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
        verdict    = "Likely Real"
        confidence = "High" if ai_percent < 15 else "Medium"
    elif ai_percent < 60:
        verdict    = "Suspicious"
        confidence = "Low"
    else:
        verdict    = "Strong AI Influence"
        confidence = "High" if ai_percent > 80 else "Medium"
    return ai_percent, verdict, confidence

# ==========================================
# 7. JSON HELPERS
# ==========================================
def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_json(filepath, data):
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"ERROR saving {filepath}: {e}")

def add_to_history(filename, file_type, ai_percent, verdict, confidence, raw_score):
    history = load_json(HISTORY_FILE)
    entry   = {
        "id":                   uuid.uuid4().hex[:8].upper(),
        "filename":             filename,
        "type":                 file_type,
        "ai_influence_percent": ai_percent,
        "raw_score":            round(raw_score, 4),
        "verdict":              verdict,
        "confidence":           confidence,
        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    history.insert(0, entry)
    history = history[:200]          # keep last 200 scans
    save_json(HISTORY_FILE, history)
    return entry

# ==========================================
# 8. PAGE ROUTES
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/batch")
def batch():
    return render_template("batch.html")

@app.route("/cybersecurity")
def cybersecurity():
    return render_template("cybersecurity.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ==========================================
# 9. PREDICT API
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    original_name = file.filename
    filename      = f"{uuid.uuid4().hex}_{original_name}"
    path          = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        is_video  = original_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        raw_score = predict_video(path) if is_video else predict_image(path)

        ai_percent, verdict, confidence = interpret_score(raw_score)
        file_type = "video" if is_video else "image"

        # Save result to history
        add_to_history(original_name, file_type, ai_percent, verdict, confidence, raw_score)

        return jsonify({
            "ai_influence_percent": ai_percent,
            "raw_score":            round(raw_score, 4),
            "type":                 file_type,
            "interpretation":       verdict,
            "confidence":           confidence,
        })

    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(path):
            os.remove(path)

# ==========================================
# 10. HISTORY API
# ==========================================
@app.route("/api/history", methods=["GET"])
def api_history():
    history   = load_json(HISTORY_FILE)
    filter_by = request.args.get("filter", "all")

    if filter_by == "real":
        history = [h for h in history if h["ai_influence_percent"] < 30]
    elif filter_by == "fake":
        history = [h for h in history if h["ai_influence_percent"] >= 60]
    elif filter_by == "suspicious":
        history = [h for h in history if 30 <= h["ai_influence_percent"] < 60]

    search = request.args.get("search", "").lower()
    if search:
        history = [h for h in history if search in h["filename"].lower()]

    total    = len(history)
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 9))
    start    = (page - 1) * per_page

    return jsonify({
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    max(1, (total + per_page - 1) // per_page),
        "results":  history[start:start + per_page],
    })


@app.route("/api/history/clear", methods=["DELETE"])
def clear_history():
    save_json(HISTORY_FILE, [])
    return jsonify({"message": "History cleared."})


@app.route("/api/history/export", methods=["GET"])
def export_history():
    history = load_json(HISTORY_FILE)
    lines   = ["ID,Filename,Type,AI Score (%),Verdict,Confidence,Timestamp"]
    for h in history:
        lines.append(
            f"{h.get('id','')},{h['filename']},{h['type']},"
            f"{h['ai_influence_percent']},{h['verdict']},{h['confidence']},{h['timestamp']}"
        )
    return Response(
        "\n".join(lines),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=checkai_history.csv"},
    )

# ==========================================
# 11. DASHBOARD API
# ==========================================
@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    from collections import defaultdict
    from datetime import timedelta

    history = load_json(HISTORY_FILE)
    total   = len(history)
    fake    = sum(1 for h in history if h["ai_influence_percent"] >= 60)
    real    = sum(1 for h in history if h["ai_influence_percent"] < 30)
    sus     = total - fake - real
    avg     = round(sum(h["ai_influence_percent"] for h in history) / total, 1) if total else 0

    daily_real = defaultdict(int)
    daily_fake = defaultdict(int)
    for h in history:
        day = h.get("timestamp", "")[:10]
        if h["ai_influence_percent"] < 30:
            daily_real[day] += 1
        elif h["ai_influence_percent"] >= 60:
            daily_fake[day] += 1

    today   = datetime.now().date()
    labels  = [(today - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
    buckets = [0, 0, 0, 0, 0]
    for h in history:
        idx = min(int(h["ai_influence_percent"] // 20), 4)
        buckets[idx] += 1

    return jsonify({
        "stats": {
            "total":      total,
            "fake":       fake,
            "real":       real,
            "suspicious": sus,
            "avg_score":  avg,
            "recent":     history[:8],
        },
        "chart": {
            "labels":    labels,
            "real_data": [daily_real.get(d, 0) for d in labels],
            "fake_data": [daily_fake.get(d, 0) for d in labels],
        },
        "buckets": buckets,
    })

# ==========================================
# 12. COMPLAINT API
# ==========================================
@app.route("/api/complaint", methods=["POST"])
def submit_complaint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        complaint_id = "CMP-" + uuid.uuid4().hex[:6].upper() + "-" + uuid.uuid4().hex[:4].upper()
        entry = {
            "complaint_id":  complaint_id,
            "name":          data.get("name", "Anonymous"),
            "email":         data.get("email", ""),
            "phone":         data.get("phone", ""),
            "category":      data.get("category", ""),
            "severity":      data.get("severity", ""),
            "incident_date": data.get("date", ""),
            "platform":      data.get("platform", ""),
            "description":   data.get("description", ""),
            "anonymous":     data.get("anonymous", False),
            "attachments":   data.get("files", []),
            "submitted_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        complaints = load_json(COMPLAINTS_FILE)
        complaints.insert(0, entry)
        save_json(COMPLAINTS_FILE, complaints)
        print(f"COMPLAINT: {complaint_id} | {entry['category']} | {entry['severity']}")

        return jsonify({
            "message":      "Complaint submitted successfully.",
            "complaint_id": complaint_id,
            "submitted_at": entry["submitted_at"],
        })

    except Exception as e:
        print(f"COMPLAINT ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/complaint/clear", methods=["DELETE"])
def clear_complaints():
    save_json(COMPLAINTS_FILE, [])
    return jsonify({"message": "Complaints cleared."})

# ==========================================
# 13. RUN  — Render injects $PORT automatically
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("=" * 50)
    print("  CheckAI — Deepfake Detection Suite")
    print(f"  Running at: http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=port)
