import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_from_directory, jsonify
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model

# ----------------- CONFIG -----------------
APP_ROOT = Path(__file__).parent.resolve()
UPLOAD_FOLDER = APP_ROOT / 'uploads'
MODEL_PATH = APP_ROOT / 'Model' / 'BrainTumorModel_MultiClass.h5'
DB_PATH = APP_ROOT / 'users.db'
ALLOWED_EXT = {'.png', '.jpg', '.jpeg'}

# Class names MUST match the HTML conditions in model_playground.html
CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

IMG_SIZE = (64, 64)  # match training size
# ------------------------------------------

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------- MODEL LOAD -----------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(str(MODEL_PATH))
print(f"✅ Model loaded from {MODEL_PATH}")

# ----------------- DB INIT -----------------
def init_db():
    with sqlite3.connect(str(DB_PATH)) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                filename TEXT,
                result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
init_db()

# ----------------- HELPERS -----------------
def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def preprocess_image(path):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError("Failed to read image.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize(IMG_SIZE)
    arr = np.array(pil_img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(path):
    x = preprocess_image(path)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    return CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "unknown"

# ----------------- ROUTES -----------------
@app.route('/')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Please fill username and password", "error")
            return render_template('login.html')

        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute('SELECT password FROM users WHERE username=?', (username,))
            row = cur.fetchone()

        if row and check_password_hash(row[0], password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password", "error")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Please provide username and password", "error")
            return render_template('register.html')

        if len(password) < 8:
            flash("Password must be at least 8 characters", "error")
            return render_template('register.html')

        hashed_pw = generate_password_hash(password)
        try:
            with sqlite3.connect(str(DB_PATH)) as conn:
                cur = conn.cursor()
                cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
                conn.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out", "info")
    return redirect(url_for('login'))

@app.route('/model', methods=['GET', 'POST'])
def model_page():
    if 'user' not in session:
        return redirect(url_for('login'))

    result, summary, preview_filename = None, None, None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file type", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = UPLOAD_FOLDER / filename

        # Avoid overwriting
        if save_path.exists():
            stem, ext = Path(filename).stem, Path(filename).suffix
            filename = f"{stem}_{int(datetime.utcnow().timestamp())}{ext}"
            save_path = UPLOAD_FOLDER / filename

        file.save(str(save_path))
        preview_filename = filename

        try:
            result = predict_image(save_path)
        except Exception as e:
            flash(f"Prediction error: {e}", "error")
            return redirect(request.url)

        # Summary text
        if result == "no_tumor":
            summary = "No tumor detected. If symptoms persist, consult a neurologist."
        elif result == "glioma_tumor":
            summary = "Glioma tumor detected. Immediate specialist evaluation recommended."
        elif result == "meningioma_tumor":
            summary = "Meningioma tumor detected. Often benign, but consult a neurosurgeon."
        elif result == "pituitary_tumor":
            summary = "Pituitary tumor detected. Endocrinology consultation advised."
        else:
            summary = f"Tumor detected: {result}. Seek medical advice."

        # Save to history
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO history (username, filename, result, timestamp) VALUES (?, ?, ?, ?)",
                (session['user'], filename, result, datetime.utcnow())
            )
            conn.commit()

    return render_template('model_playground.html', result=result, summary=summary, preview=preview_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    with sqlite3.connect(str(DB_PATH)) as conn:
        cur = conn.cursor()
        cur.execute(
            'SELECT filename, result, timestamp FROM history WHERE username=? ORDER BY timestamp DESC',
            (session['user'],)
        )
        rows = cur.fetchall()

    history_list = [
        {'filename': r[0], 'result': r[1], 'timestamp': r[2]} for r in rows
    ]
    return render_template('history.html', history=history_list)

@app.route('/api/history/<int:n>')
def api_history(n=10):
    if 'user' not in session:
        return {"error": "unauthenticated"}, 401
    with sqlite3.connect(str(DB_PATH)) as conn:
        cur = conn.cursor()
        cur.execute(
            'SELECT filename, result, timestamp FROM history WHERE username=? ORDER BY timestamp DESC LIMIT ?',
            (session['user'], n)
        )
        rows = cur.fetchall()
    return jsonify([{'filename': r[0], 'result': r[1], 'timestamp': r[2]} for r in rows])

# ----------------- RUN -----------------
if __name__ == '__main__':
    app.run(debug=True)
