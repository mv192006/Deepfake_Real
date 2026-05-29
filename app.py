import os
import cv2
import numpy as np
from flask import Flask, render_template_string, request, redirect, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model
model = load_model("deepfake_model.h5")

# NOTE: Set to True if your model outputs 1=FAKE, 0=REAL (most public deepfake datasets).
# Flip to False if 1=REAL, 0=FAKE. If results seem inverted, toggle this flag.
MODEL_FAKE_IS_HIGH = False

# Haar cascade for face detection (ships with OpenCV, no extra install needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def allowed_file(name, file_type=None):
    if '.' not in name:
        return False
    ext = name.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in {'png', 'jpg', 'jpeg'}
    elif file_type == 'video':
        return ext in {'mp4', 'avi', 'mov', 'mkv'}
    return ext in ALLOWED_EXTENSIONS

# ---------------- ML LOGIC ----------------

def extract_face(frame_bgr, target_size=(224, 224), padding=0.2):
    """
    Detect the largest face in a BGR frame, apply padding, convert to RGB,
    resize, and return a normalized float32 array shaped (1, H, W, 3).
    Returns None if no face is found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    # Pick the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add padding around the face crop
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_bgr.shape[1], x + w + pad_x)
    y2 = min(frame_bgr.shape[0], y + h + pad_y)

    face_crop = frame_bgr[y1:y2, x1:x2]

    # BGR -> RGB (Keras models expect RGB)
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, target_size)
    face_array = img_to_array(face_resized) / 255.0
    return np.expand_dims(face_array, axis=0)


def score_to_label(score):
    """
    Convert raw model score (0-1) to (label, confidence_pct).
    Respects the MODEL_FAKE_IS_HIGH flag.
    """
    if MODEL_FAKE_IS_HIGH:
        if score > 0.5:
            return "FAKE", score * 100
        else:
            return "REAL", (1 - score) * 100
    else:
        if score > 0.5:
            return "REAL", score * 100
        else:
            return "FAKE", (1 - score) * 100


def predict_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError("Could not read image file.")

    face = extract_face(frame)

    if face is None:
        # No face detected — fall back to full-image with BGR->RGB fix
        print("[WARN] No face detected in image, falling back to full-image prediction.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        face = np.expand_dims(img_to_array(resized) / 255.0, axis=0)

    score = float(model.predict(face, verbose=0)[0][0])
    print(f"[IMAGE] raw score={score:.4f}")
    label, confidence = score_to_label(score)
    return label, confidence


def predict_video(path, max_frames=30):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0, []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, frame_count // max_frames)

    predictions = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            face = extract_face(frame)
            if face is not None:
                score = float(model.predict(face, verbose=0)[0][0])
                predictions.append(score)
                print(f"[VIDEO] frame {frame_idx}: score={score:.4f}")
            else:
                print(f"[VIDEO] frame {frame_idx}: no face detected, skipped")

        frame_idx += 1

    cap.release()

    if not predictions:
        print("[WARN] No faces detected in any sampled video frame.")
        return "UNKNOWN", 0, []

    avg = float(np.mean(predictions))
    print(f"[VIDEO] avg={avg:.4f} over {len(predictions)} face frames")
    label, confidence = score_to_label(avg)
    return label, confidence, predictions

# ---------------- STATIC FILES ----------------

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# ---------------- HTML PAGES ----------------

LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Abridge AI - Deepfake Detection Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: #0a0e27; color: #ffffff; overflow-x: hidden; }
        nav { display: flex; justify-content: space-between; align-items: center; padding: 20px 60px; background: rgba(10,14,39,0.95); backdrop-filter: blur(10px); position: sticky; top: 0; z-index: 100; border-bottom: 1px solid rgba(0,194,203,0.1); }
        .logo { display: flex; align-items: center; gap: 12px; font-size: 1.5rem; font-weight: 700; }
        .logo-icon { width: 80px; height: 80px; }
        .logo-icon img { width: 100%; height: 100%; object-fit: contain; }
        .nav-links { display: flex; gap: 40px; list-style: none; }
        .nav-links a { color: #ffffff; text-decoration: none; font-size: 1rem; transition: color 0.3s; }
        .nav-links a:hover { color: #00c2cb; }
        .nav-cta { padding: 12px 28px; background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); border-radius: 50px; text-decoration: none; color: white; font-weight: 600; transition: transform 0.3s, box-shadow 0.3s; }
        .nav-cta:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,194,203,0.3); }
        .hero { text-align: center; padding: 120px 40px 80px; position: relative; }
        .trust-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px; background: rgba(0,194,203,0.1); border: 1px solid rgba(0,194,203,0.3); border-radius: 50px; font-size: 0.9rem; margin-bottom: 30px; color: #00c2cb; }
        .hero h1 { font-size: 4.5rem; font-weight: 800; line-height: 1.1; margin-bottom: 20px; }
        .gradient-text { background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .hero p { font-size: 1.4rem; color: #a0a8c5; max-width: 800px; margin: 0 auto 40px; line-height: 1.8; }
        .hero-buttons { display: flex; gap: 20px; justify-content: center; margin-top: 40px; }
        .btn-primary { padding: 18px 40px; background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); border-radius: 50px; text-decoration: none; color: white; font-weight: 700; font-size: 1.1rem; transition: all 0.3s; display: inline-flex; align-items: center; gap: 10px; }
        .btn-primary:hover { transform: translateY(-3px); box-shadow: 0 15px 40px rgba(0,194,203,0.4); }
        .btn-secondary { padding: 18px 40px; background: rgba(255,255,255,0.05); border: 2px solid rgba(255,255,255,0.1); border-radius: 50px; text-decoration: none; color: white; font-weight: 700; font-size: 1.1rem; transition: all 0.3s; }
        .btn-secondary:hover { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.2); }
        .problem-section { padding: 100px 60px; background: rgba(255,255,255,0.02); }
        .section-title { text-align: center; font-size: 2.8rem; font-weight: 800; margin-bottom: 60px; }
        .problem-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 40px; max-width: 1200px; margin: 0 auto; }
        .problem-card { padding: 40px; background: rgba(255,255,255,0.03); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); transition: all 0.3s; }
        .problem-card:hover { background: rgba(255,255,255,0.05); border-color: rgba(0,194,203,0.3); transform: translateY(-5px); }
        .problem-card h3 { font-size: 1.5rem; margin-bottom: 15px; color: #00c2cb; }
        .problem-card p { color: #a0a8c5; line-height: 1.6; }
        .features-section { padding: 100px 60px; }
        .features-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 40px; max-width: 1200px; margin: 0 auto; }
        .feature-card { padding: 40px; background: linear-gradient(135deg, rgba(0,194,203,0.05) 0%, rgba(0,119,182,0.05) 100%); border-radius: 20px; border: 1px solid rgba(0,194,203,0.2); }
        .feature-icon { font-size: 3rem; margin-bottom: 20px; }
        .feature-card h3 { font-size: 1.5rem; margin-bottom: 15px; }
        .feature-card ul { list-style: none; color: #a0a8c5; }
        .feature-card li { padding: 8px 0; padding-left: 25px; position: relative; }
        .feature-card li:before { content: "✓"; position: absolute; left: 0; color: #00c2cb; font-weight: bold; }
        .how-it-works { padding: 100px 60px; background: rgba(255,255,255,0.02); }
        .steps-container { display: grid; grid-template-columns: repeat(3,1fr); gap: 60px; max-width: 1200px; margin: 60px auto 0; }
        .step { text-align: center; }
        .step-number { width: 60px; height: 60px; background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: 800; margin: 0 auto 20px; }
        .step h3 { font-size: 1.5rem; margin-bottom: 15px; }
        .step p { color: #a0a8c5; line-height: 1.6; }
        .pricing-section { padding: 100px 60px; }
        .pricing-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 40px; max-width: 1200px; margin: 60px auto 0; }
        .pricing-card { padding: 50px 40px; background: rgba(255,255,255,0.03); border-radius: 24px; border: 1px solid rgba(255,255,255,0.05); transition: all 0.3s; }
        .pricing-card.featured { background: linear-gradient(135deg, rgba(0,194,203,0.1) 0%, rgba(0,119,182,0.1) 100%); border: 2px solid #00c2cb; transform: scale(1.05); }
        .pricing-card:hover { transform: translateY(-5px) scale(1.02); }
        .pricing-card.featured:hover { transform: translateY(-5px) scale(1.07); }
        .pricing-title { font-size: 1.5rem; margin-bottom: 10px; }
        .pricing-subtitle { color: #a0a8c5; margin-bottom: 30px; }
        .pricing-price { font-size: 3rem; font-weight: 800; margin-bottom: 30px; }
        .pricing-features { list-style: none; margin-bottom: 30px; }
        .pricing-features li { padding: 12px 0; color: #a0a8c5; }
        .pricing-btn { width: 100%; padding: 15px; background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); border: none; border-radius: 50px; color: white; font-weight: 700; font-size: 1.1rem; cursor: pointer; transition: all 0.3s; }
        .pricing-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,194,203,0.3); }
        .team-section { padding: 100px 60px; background: rgba(255,255,255,0.02); }
        .team-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 50px; max-width: 1000px; margin: 60px auto 0; }
        .team-member { text-align: center; }
        .team-avatar { width: 120px; height: 120px; background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center; font-size: 2.5rem; font-weight: 800; }
        .team-name { font-size: 1.3rem; font-weight: 700; margin-bottom: 5px; }
        .team-role { color: #00c2cb; font-weight: 600; margin-bottom: 10px; }
        .team-description { color: #a0a8c5; font-size: 0.95rem; }
        .contact-section { padding: 100px 60px; }
        .contact-container { max-width: 800px; margin: 0 auto; }
        .contact-info { display: grid; grid-template-columns: repeat(3,1fr); gap: 40px; margin-bottom: 60px; }
        .contact-item { text-align: center; }
        .contact-item h3 { font-size: 1.2rem; margin-bottom: 10px; color: #00c2cb; }
        .contact-item p { color: #a0a8c5; }
        .contact-form { display: grid; gap: 20px; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        input, textarea { width: 100%; padding: 15px 20px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; color: white; font-size: 1rem; }
        input:focus, textarea:focus { outline: none; border-color: #00c2cb; background: rgba(255,255,255,0.08); }
        textarea { resize: vertical; min-height: 150px; }
        footer { padding: 60px; background: rgba(0,0,0,0.3); border-top: 1px solid rgba(255,255,255,0.05); }
        .footer-content { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 60px; max-width: 1200px; margin: 0 auto 40px; }
        .footer-brand { display: flex; align-items: center; gap: 12px; font-size: 1.5rem; font-weight: 700; margin-bottom: 15px; }
        .footer-brand .logo-icon { width: 80px; height: 80px; }
        .footer-description { color: #a0a8c5; line-height: 1.6; }
        .footer-section h4 { margin-bottom: 20px; font-size: 1.1rem; }
        .footer-links { list-style: none; }
        .footer-links li { margin-bottom: 12px; }
        .footer-links a { color: #a0a8c5; text-decoration: none; transition: color 0.3s; }
        .footer-links a:hover { color: #00c2cb; }
        .footer-bottom { text-align: center; padding-top: 40px; border-top: 1px solid rgba(255,255,255,0.05); color: #a0a8c5; }
        @media (max-width: 968px) {
            .problem-grid, .features-grid, .steps-container, .pricing-grid, .team-grid { grid-template-columns: 1fr; }
            .hero h1 { font-size: 3rem; }
            .footer-content { grid-template-columns: 1fr; }
            nav { padding: 20px 30px; }
            .nav-links { display: none; }
        }
    </style>
</head>
<body>
    <nav>
        <div class="logo">
            <div class="logo-icon"><img src="/static/logo.png" alt="Abridge AI Logo"></div>
            <span>Abridge AI</span>
        </div>
        <ul class="nav-links">
            <li><a href="#features">Features</a></li>
            <li><a href="#how-it-works">How It Works</a></li>
            <li><a href="#pricing">Pricing</a></li>
            <li><a href="#contact">Contact</a></li>
        </ul>
        <a href="/detect" class="nav-cta">Try Demo</a>
    </nav>

    <section class="hero">
        <div class="trust-badge"><span>⭐</span> Trusted by media & enterprises worldwide</div>
        <h1>Defending Reality.<br><span class="gradient-text">Restoring Trust.</span></h1>
        <p>Abridge AI detects deepfakes in real-time across videos, images, and audio. Verify authenticity. Restore confidence. Protect your digital future.</p>
        <div class="hero-buttons">
            <a href="/detect" class="btn-primary">Try Demo →</a>
            <a href="#how-it-works" class="btn-secondary">View Documentation</a>
        </div>
    </section>

    <section class="problem-section">
        <h2 class="section-title">Deepfakes are breaking digital trust.</h2>
        <p style="text-align:center;color:#a0a8c5;max-width:800px;margin:0 auto 60px;font-size:1.2rem;">In a world where synthetic media can be weaponized, verification is no longer optional.</p>
        <div class="problem-grid">
            <div class="problem-card"><h3>Deepfakes Break Trust</h3><p>Synthetic media is becoming indistinguishable from reality. Misinformation spreads faster than truth.</p></div>
            <div class="problem-card"><h3>Financial & Legal Risks</h3><p>Banks, media, and institutions face fraud, reputation damage, and regulatory compliance challenges.</p></div>
            <div class="problem-card"><h3>Manual Verification Fails</h3><p>Human review is slow, costly, and often unreliable. Real-time detection is impossible at scale.</p></div>
        </div>
    </section>

    <section class="features-section" id="features">
        <h2 class="section-title">How Abridge AI Protects You</h2>
        <p style="text-align:center;color:#a0a8c5;max-width:800px;margin:0 auto 60px;font-size:1.2rem;">See the truth behind every piece of digital media with advanced AI detection.</p>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3>AI-Powered Detection</h3>
                <p style="color:#a0a8c5;margin-bottom:20px;">Advanced neural networks trained on millions of samples. Detects subtle manipulations invisible to human eyes.</p>
                <ul><li>Video deepfakes</li><li>Image manipulation</li><li>Voice cloning</li></ul>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Real-Time Verification</h3>
                <p style="color:#a0a8c5;margin-bottom:20px;">Analyze content in seconds. API-first architecture scales to enterprise demands without compromising speed.</p>
                <ul><li>&lt; 60 second analysis</li><li>Enterprise-grade uptime</li><li>99% reliability</li></ul>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Confidence Scores</h3>
                <p style="color:#a0a8c5;margin-bottom:20px;">Transparent, interpretable results. Understand exactly what triggered detection and confidence levels.</p>
                <ul><li>Detailed reports</li><li>Audit trails</li><li>Explainable AI</li></ul>
            </div>
        </div>
    </section>

    <section class="how-it-works" id="how-it-works">
        <h2 class="section-title">How It Works</h2>
        <p style="text-align:center;color:#a0a8c5;max-width:800px;margin:0 auto;font-size:1.2rem;">Three simple steps to verify authenticity</p>
        <div class="steps-container">
            <div class="step"><div class="step-number">01</div><h3>Upload Content</h3><p>Submit video, image, or audio file through our web interface or API.</p></div>
            <div class="step"><div class="step-number">02</div><h3>AI Analysis</h3><p>Our neural networks analyze the content for signs of manipulation and deepfake indicators.</p></div>
            <div class="step"><div class="step-number">03</div><h3>Get Results</h3><p>Receive detailed report with confidence scores and actionable insights within seconds.</p></div>
        </div>
    </section>

    <section class="pricing-section" id="pricing">
        <h2 class="section-title">Simple, Transparent Pricing</h2>
        <p style="text-align:center;color:#a0a8c5;max-width:800px;margin:0 auto;font-size:1.2rem;">Choose the plan that scales with your needs</p>
        <div class="pricing-grid">
            <div class="pricing-card">
                <div class="pricing-title">Starter</div>
                <div class="pricing-subtitle">Perfect for individuals and small teams</div>
                <div class="pricing-price">₹99<span style="font-size:1.2rem;font-weight:400;">/image</span></div>
                <ul class="pricing-features">
                    <li>✓ Basic image/video forgery checks</li><li>✓ Web dashboard</li><li>✓ Email support</li><li>✓ Standard models</li>
                </ul>
                <button class="pricing-btn" onclick="window.location.href='/detect'">Get Started</button>
            </div>
            <div class="pricing-card featured">
                <div class="pricing-title">Professional</div>
                <div class="pricing-subtitle">For growing media companies</div>
                <div class="pricing-price">₹999<span style="font-size:1.2rem;font-weight:400;">/month</span></div>
                <ul class="pricing-features">
                    <li>✓ Priority support</li><li>✓ Advanced analytics</li><li>✓ Custom integrations</li><li>✓ Full access to all features</li>
                </ul>
                <button class="pricing-btn" onclick="window.location.href='/detect'">Start Free Trial</button>
            </div>
            <div class="pricing-card">
                <div class="pricing-title">Enterprise</div>
                <div class="pricing-subtitle">For large-scale operations</div>
                <div class="pricing-price">Custom</div>
                <ul class="pricing-features">
                    <li>✓ White-label solution</li><li>✓ Custom models</li><li>✓ 24/7 support</li><li>✓ SLA guarantee</li><li>✓ On-premise option</li>
                </ul>
                <button class="pricing-btn">Contact Sales</button>
            </div>
        </div>
    </section>

    <section class="team-section" id="team">
        <h2 class="section-title">Meet the Team</h2>
        <p style="text-align:center;color:#a0a8c5;max-width:800px;margin:0 auto;font-size:1.2rem;">Built by experts in AI, security, and enterprise tech</p>
        <div class="team-grid">
            <div class="team-member"><div class="team-avatar">MV</div><div class="team-name">Mukul Varade</div><div class="team-role">CEO & Co-Founder</div><div class="team-description">Manages project direction, coordination, and overall planning.</div></div>
            <div class="team-member"><div class="team-avatar">AP</div><div class="team-name">Arjun Patil</div><div class="team-role">CFO & Co-Founder</div><div class="team-description">Responsible for financial strategy and business operations.</div></div>
            <div class="team-member"><div class="team-avatar">AF</div><div class="team-name">Atharv Fatale</div><div class="team-role">CTO & Co-Founder</div><div class="team-description">Handles the technical development and AI model work.</div></div>
        </div>
    </section>

    <section class="contact-section" id="contact">
        <h2 class="section-title">Get in Touch</h2>
        <p style="text-align:center;color:#a0a8c5;max-width:800px;margin:0 auto 60px;font-size:1.2rem;">Have questions? Our team is here to help.</p>
        <div class="contact-container">
            <div class="contact-info">
                <div class="contact-item"><h3>Email</h3><p>contact@abridgeai.com</p></div>
                <div class="contact-item"><h3>Phone</h3><p>+91 91234 56789</p></div>
                <div class="contact-item"><h3>Office</h3><p>Pune, Maharashtra</p></div>
            </div>
            <form class="contact-form" onsubmit="return false;">
                <div class="form-row">
                    <input type="text" placeholder="Full Name" required>
                    <input type="email" placeholder="Email" required>
                </div>
                <input type="text" placeholder="Company">
                <textarea placeholder="Message" required></textarea>
                <button type="submit" class="btn-primary" style="width:100%;justify-content:center;border:none;cursor:pointer;">Send Message</button>
            </form>
        </div>
    </section>

    <footer>
        <div class="footer-content">
            <div>
                <div class="footer-brand"><div class="logo-icon"><img src="/static/logo.png" alt="Abridge AI Logo"></div><span>Abridge AI</span></div>
                <p class="footer-description">Defending reality in the age of deepfakes.</p>
            </div>
            <div class="footer-section"><h4>Product</h4><ul class="footer-links"><li><a href="#features">Features</a></li><li><a href="#pricing">Pricing</a></li><li><a href="/detect">API Docs</a></li><li><a href="#">Blog</a></li></ul></div>
            <div class="footer-section"><h4>Company</h4><ul class="footer-links"><li><a href="#">About</a></li><li><a href="#team">Team</a></li><li><a href="#">Careers</a></li><li><a href="#contact">Contact</a></li></ul></div>
            <div class="footer-section"><h4>Legal</h4><ul class="footer-links"><li><a href="#">Privacy</a></li><li><a href="#">Terms</a></li><li><a href="#">Security</a></li><li><a href="#">Compliance</a></li></ul></div>
        </div>
        <div class="footer-bottom">© 2026 Abridge AI. All rights reserved.</div>
    </footer>
</body>
</html>
"""

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Dashboard - Abridge AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0e27; color: white; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 20px; }
        nav { width: 100%; max-width: 1200px; display: flex; justify-content: space-between; align-items: center; padding: 20px 0; margin-bottom: 40px; }
        .logo { display: flex; align-items: center; gap: 12px; font-size: 1.5rem; font-weight: 700; }
        .logo-icon { width: 80px; height: 80px; }
        .logo-icon img { width: 100%; height: 100%; object-fit: contain; }
        .back-link { color: #00c2cb; text-decoration: none; transition: opacity 0.3s; }
        .back-link:hover { opacity: 0.8; }
        .container { max-width: 1200px; width: 100%; }
        .hero { text-align: center; margin-bottom: 60px; }
        .hero h1 { font-size: 3.5rem; font-weight: 800; margin-bottom: 15px; }
        .gradient-text { background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero p { font-size: 1.3rem; color: #a0a8c5; }
        .cards-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 40px; }
        .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,194,203,0.2); border-radius: 24px; padding: 50px 40px; text-align: center; transition: all 0.3s; cursor: pointer; }
        .card:hover { background: rgba(255,255,255,0.08); border-color: #00c2cb; transform: translateY(-10px); box-shadow: 0 20px 60px rgba(0,194,203,0.3); }
        .card-icon { font-size: 4rem; margin-bottom: 25px; }
        .card h2 { font-size: 2rem; margin-bottom: 15px; }
        .card p { font-size: 1.1rem; color: #a0a8c5; margin-bottom: 30px; line-height: 1.6; }
        .btn { display: inline-block; padding: 16px 40px; background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); color: white; text-decoration: none; border-radius: 50px; font-size: 1.1rem; font-weight: 600; transition: all 0.3s; border: none; cursor: pointer; }
        .btn:hover { transform: scale(1.05); box-shadow: 0 10px 30px rgba(0,194,203,0.4); }
        @media (max-width: 768px) { .hero h1 { font-size: 2.5rem; } .cards-container { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <nav>
        <div class="logo"><div class="logo-icon"><img src="/static/logo.png" alt="Abridge AI Logo"></div><span>Abridge AI</span></div>
        <a href="/" class="back-link">← Back to Home</a>
    </nav>
    <div class="container">
        <div class="hero">
            <h1><span class="gradient-text">Detection Dashboard</span></h1>
            <p>Choose your analysis type</p>
        </div>
        <div class="cards-container">
            <div class="card" onclick="window.location.href='/image'">
                <div class="card-icon">🖼️</div>
                <h2>Image Analysis</h2>
                <p>Upload an image to detect if it has been manipulated using deepfake technology</p>
                <button class="btn">Analyze Image</button>
            </div>
            <div class="card" onclick="window.location.href='/video'">
                <div class="card-icon">🎥</div>
                <h2>Video Analysis</h2>
                <p>Upload a video for frame-by-frame analysis to detect deepfake manipulation</p>
                <button class="btn">Analyze Video</button>
            </div>
        </div>
    </div>
</body>
</html>
"""

IMAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Analysis - Abridge AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0e27; color: white; min-height: 100vh; padding: 20px; }
        nav { max-width: 900px; margin: 0 auto 40px; display: flex; justify-content: space-between; align-items: center; padding: 20px 0; }
        .logo { display: flex; align-items: center; gap: 12px; font-size: 1.5rem; font-weight: 700; }
        .logo-icon { width: 80px; height: 80px; }
        .logo-icon img { width: 100%; height: 100%; object-fit: contain; }
        .back-link { color: #00c2cb; text-decoration: none; transition: opacity 0.3s; }
        .back-link:hover { opacity: 0.8; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .gradient-text { background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { color: #a0a8c5; font-size: 1.1rem; }
        .upload-container { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,194,203,0.2); border-radius: 24px; padding: 50px; }
        .upload-box { border: 3px dashed rgba(0,194,203,0.3); border-radius: 16px; padding: 60px 20px; text-align: center; transition: all 0.3s; cursor: pointer; background: rgba(0,194,203,0.05); }
        .upload-box:hover { border-color: #00c2cb; background: rgba(0,194,203,0.1); }
        .upload-box.dragover { border-color: #00c2cb; background: rgba(0,194,203,0.15); transform: scale(1.02); }
        .upload-icon { font-size: 4rem; margin-bottom: 20px; }
        .upload-text { font-size: 1.3rem; margin-bottom: 10px; }
        .upload-subtext { font-size: 1rem; color: #a0a8c5; }
        input[type="file"] { display: none; }
        .preview-container { margin-top: 30px; display: none; }
        .preview-image { max-width: 100%; border-radius: 12px; margin-bottom: 20px; border: 2px solid rgba(0,194,203,0.2); }
        .analyze-btn { width: 100%; padding: 18px; background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); color: white; border: none; border-radius: 50px; font-size: 1.2rem; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .analyze-btn:hover { transform: scale(1.02); box-shadow: 0 10px 30px rgba(0,194,203,0.4); }
        .analyze-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .loader { display: none; text-align: center; margin-top: 30px; }
        .spinner { border: 4px solid rgba(255,255,255,0.1); border-top: 4px solid #00c2cb; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .result-container { display: none; margin-top: 30px; padding: 30px; border-radius: 16px; text-align: center; border: 2px solid; }
        .result-real { background: rgba(34,197,94,0.1); border-color: #22c55e; }
        .result-fake { background: rgba(239,68,68,0.1); border-color: #ef4444; }
        .result-unknown { background: rgba(234,179,8,0.1); border-color: #eab308; }
        .result-title { font-size: 2rem; font-weight: 700; margin-bottom: 15px; }
        .result-real .result-title { color: #22c55e; }
        .result-fake .result-title { color: #ef4444; }
        .result-unknown .result-title { color: #eab308; }
        .result-confidence { font-size: 1.3rem; margin-bottom: 10px; color: #a0a8c5; }
        .result-note { font-size: 0.95rem; color: #a0a8c5; margin-bottom: 20px; font-style: italic; }
        .confidence-bar-wrap { background: rgba(255,255,255,0.1); border-radius: 50px; height: 10px; margin: 10px 0 20px; overflow: hidden; }
        .confidence-bar { height: 100%; border-radius: 50px; transition: width 0.8s ease; }
        .reset-btn { padding: 12px 30px; background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2); border-radius: 50px; font-size: 1rem; cursor: pointer; transition: all 0.3s; }
        .reset-btn:hover { background: rgba(255,255,255,0.15); }
    </style>
</head>
<body>
    <nav>
        <div class="logo"><div class="logo-icon"><img src="/static/logo.png" alt="Abridge AI Logo"></div><span>Abridge AI</span></div>
        <a href="/detect" class="back-link">← Back to Dashboard</a>
    </nav>
    <div class="container">
        <div class="header">
            <h1><span class="gradient-text">Image Analysis</span></h1>
            <p>Upload an image to detect deepfake manipulation</p>
        </div>
        <div class="upload-container">
            <div class="upload-box" id="uploadBox">
                <div class="upload-icon">📤</div>
                <div class="upload-text">Drag & Drop your image here</div>
                <div class="upload-subtext">or click to browse (PNG, JPG, JPEG)</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            <div class="preview-container" id="previewContainer">
                <img id="previewImage" class="preview-image" src="" alt="Preview">
                <button class="analyze-btn" id="analyzeBtn">Analyze Image</button>
            </div>
            <div class="loader" id="loader">
                <div class="spinner"></div>
                <p>Analyzing image... Please wait</p>
            </div>
            <div class="result-container" id="resultContainer">
                <div class="result-title" id="resultTitle"></div>
                <div class="confidence-bar-wrap"><div class="confidence-bar" id="confidenceBar"></div></div>
                <div class="result-confidence" id="resultConfidence"></div>
                <div class="result-note" id="resultNote"></div>
                <button class="reset-btn" onclick="resetAnalysis()">Analyze Another Image</button>
            </div>
        </div>
    </div>
    <script>
        const uploadBox = document.getElementById('uploadBox');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const loader = document.getElementById('loader');
        const resultContainer = document.getElementById('resultContainer');
        const resultTitle = document.getElementById('resultTitle');
        const resultConfidence = document.getElementById('resultConfidence');
        const resultNote = document.getElementById('resultNote');
        const confidenceBar = document.getElementById('confidenceBar');
        let selectedFile = null;

        uploadBox.addEventListener('click', () => fileInput.click());
        uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.classList.add('dragover'); });
        uploadBox.addEventListener('dragleave', () => { uploadBox.classList.remove('dragover'); });
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault(); uploadBox.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length > 0) handleFile(e.target.files[0]); });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) { alert('Please upload an image file'); return; }
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewContainer.style.display = 'block';
                uploadBox.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            const formData = new FormData();
            formData.append('file', selectedFile);
            analyzeBtn.disabled = true;
            loader.style.display = 'block';
            resultContainer.style.display = 'none';
            try {
                const response = await fetch('/analyze_image', { method: 'POST', body: formData });
                const data = await response.json();
                loader.style.display = 'none';
                if (data.success) {
                    const isFake = data.result === 'FAKE';
                    const isUnknown = data.result === 'UNKNOWN';
                    resultTitle.textContent = isUnknown ? '⚠ UNKNOWN' : (isFake ? '🚨 FAKE DETECTED' : '✅ REAL');
                    resultConfidence.textContent = 'Confidence: ' + data.confidence.toFixed(2) + '%';
                    resultNote.textContent = data.no_face
                        ? 'No face was detected — result is based on full-image analysis and may be less accurate.'
                        : '';
                    confidenceBar.style.width = data.confidence.toFixed(1) + '%';
                    confidenceBar.style.background = isFake
                        ? 'linear-gradient(90deg,#ef4444,#dc2626)'
                        : isUnknown
                            ? 'linear-gradient(90deg,#eab308,#ca8a04)'
                            : 'linear-gradient(90deg,#22c55e,#16a34a)';
                    resultContainer.className = 'result-container ' + (isFake ? 'result-fake' : isUnknown ? 'result-unknown' : 'result-real');
                    resultContainer.style.display = 'block';
                } else {
                    alert('Error: ' + data.message);
                    analyzeBtn.disabled = false;
                }
            } catch (error) {
                loader.style.display = 'none';
                alert('Error: ' + error.message);
                analyzeBtn.disabled = false;
            }
        });

        function resetAnalysis() {
            selectedFile = null; fileInput.value = '';
            previewContainer.style.display = 'none';
            uploadBox.style.display = 'block';
            resultContainer.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    </script>
</body>
</html>
"""

VIDEO_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analysis - Abridge AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0e27; color: white; min-height: 100vh; padding: 20px; }
        nav { max-width: 900px; margin: 0 auto 40px; display: flex; justify-content: space-between; align-items: center; padding: 20px 0; }
        .logo { display: flex; align-items: center; gap: 12px; font-size: 1.5rem; font-weight: 700; }
        .logo-icon { width: 80px; height: 80px; }
        .logo-icon img { width: 100%; height: 100%; object-fit: contain; }
        .back-link { color: #00c2cb; text-decoration: none; transition: opacity 0.3s; }
        .back-link:hover { opacity: 0.8; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .gradient-text { background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { color: #a0a8c5; font-size: 1.1rem; }
        .upload-container { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,194,203,0.2); border-radius: 24px; padding: 50px; }
        .upload-box { border: 3px dashed rgba(0,194,203,0.3); border-radius: 16px; padding: 60px 20px; text-align: center; transition: all 0.3s; cursor: pointer; background: rgba(0,194,203,0.05); }
        .upload-box:hover { border-color: #00c2cb; background: rgba(0,194,203,0.1); }
        .upload-box.dragover { border-color: #00c2cb; background: rgba(0,194,203,0.15); transform: scale(1.02); }
        .upload-icon { font-size: 4rem; margin-bottom: 20px; }
        .upload-text { font-size: 1.3rem; margin-bottom: 10px; }
        .upload-subtext { font-size: 1rem; color: #a0a8c5; }
        input[type="file"] { display: none; }
        .preview-container { margin-top: 30px; display: none; }
        .preview-video { width: 100%; border-radius: 12px; margin-bottom: 20px; border: 2px solid rgba(0,194,203,0.2); }
        .analyze-btn { width: 100%; padding: 18px; background: linear-gradient(135deg, #00c2cb 0%, #0077b6 100%); color: white; border: none; border-radius: 50px; font-size: 1.2rem; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .analyze-btn:hover { transform: scale(1.02); box-shadow: 0 10px 30px rgba(0,194,203,0.4); }
        .analyze-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .loader { display: none; text-align: center; margin-top: 30px; }
        .spinner { border: 4px solid rgba(255,255,255,0.1); border-top: 4px solid #00c2cb; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .result-container { display: none; margin-top: 30px; padding: 30px; border-radius: 16px; text-align: center; border: 2px solid; }
        .result-real { background: rgba(34,197,94,0.1); border-color: #22c55e; }
        .result-fake { background: rgba(239,68,68,0.1); border-color: #ef4444; }
        .result-unknown { background: rgba(234,179,8,0.1); border-color: #eab308; }
        .result-title { font-size: 2rem; font-weight: 700; margin-bottom: 15px; }
        .result-real .result-title { color: #22c55e; }
        .result-fake .result-title { color: #ef4444; }
        .result-unknown .result-title { color: #eab308; }
        .result-confidence { font-size: 1.3rem; margin-bottom: 10px; color: #a0a8c5; }
        .result-details { font-size: 1rem; color: #a0a8c5; margin-bottom: 8px; }
        .result-note { font-size: 0.95rem; color: #a0a8c5; margin-bottom: 20px; font-style: italic; }
        .confidence-bar-wrap { background: rgba(255,255,255,0.1); border-radius: 50px; height: 10px; margin: 10px 0 20px; overflow: hidden; }
        .confidence-bar { height: 100%; border-radius: 50px; transition: width 0.8s ease; }
        .reset-btn { padding: 12px 30px; background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2); border-radius: 50px; font-size: 1rem; cursor: pointer; transition: all 0.3s; }
        .reset-btn:hover { background: rgba(255,255,255,0.15); }
    </style>
</head>
<body>
    <nav>
        <div class="logo"><div class="logo-icon"><img src="/static/logo.png" alt="Abridge AI Logo"></div><span>Abridge AI</span></div>
        <a href="/detect" class="back-link">← Back to Dashboard</a>
    </nav>
    <div class="container">
        <div class="header">
            <h1><span class="gradient-text">Video Analysis</span></h1>
            <p>Upload a video for frame-by-frame deepfake detection</p>
        </div>
        <div class="upload-container">
            <div class="upload-box" id="uploadBox">
                <div class="upload-icon">📤</div>
                <div class="upload-text">Drag & Drop your video here</div>
                <div class="upload-subtext">or click to browse (MP4, AVI, MOV, MKV)</div>
                <input type="file" id="fileInput" accept="video/*">
            </div>
            <div class="preview-container" id="previewContainer">
                <video id="previewVideo" class="preview-video" controls>
                    <source id="videoSource" src="" type="video/mp4">
                </video>
                <button class="analyze-btn" id="analyzeBtn">Analyze Video</button>
            </div>
            <div class="loader" id="loader">
                <div class="spinner"></div>
                <p>Analyzing video frames... This may take a while</p>
            </div>
            <div class="result-container" id="resultContainer">
                <div class="result-title" id="resultTitle"></div>
                <div class="confidence-bar-wrap"><div class="confidence-bar" id="confidenceBar"></div></div>
                <div class="result-confidence" id="resultConfidence"></div>
                <div class="result-details" id="resultDetails"></div>
                <div class="result-note" id="resultNote"></div>
                <button class="reset-btn" onclick="resetAnalysis()">Analyze Another Video</button>
            </div>
        </div>
    </div>
    <script>
        const uploadBox = document.getElementById('uploadBox');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewVideo = document.getElementById('previewVideo');
        const videoSource = document.getElementById('videoSource');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const loader = document.getElementById('loader');
        const resultContainer = document.getElementById('resultContainer');
        const resultTitle = document.getElementById('resultTitle');
        const resultConfidence = document.getElementById('resultConfidence');
        const resultDetails = document.getElementById('resultDetails');
        const resultNote = document.getElementById('resultNote');
        const confidenceBar = document.getElementById('confidenceBar');
        let selectedFile = null;

        uploadBox.addEventListener('click', () => fileInput.click());
        uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.classList.add('dragover'); });
        uploadBox.addEventListener('dragleave', () => { uploadBox.classList.remove('dragover'); });
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault(); uploadBox.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length > 0) handleFile(e.target.files[0]); });

        function handleFile(file) {
            if (!file.type.startsWith('video/')) { alert('Please upload a video file'); return; }
            selectedFile = file;
            const url = URL.createObjectURL(file);
            videoSource.src = url;
            previewVideo.load();
            previewContainer.style.display = 'block';
            uploadBox.style.display = 'none';
        }

        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            const formData = new FormData();
            formData.append('file', selectedFile);
            analyzeBtn.disabled = true;
            loader.style.display = 'block';
            resultContainer.style.display = 'none';
            try {
                const response = await fetch('/analyze_video', { method: 'POST', body: formData });
                const data = await response.json();
                loader.style.display = 'none';
                if (data.success) {
                    const isFake = data.result === 'FAKE';
                    const isUnknown = data.result === 'UNKNOWN';
                    resultTitle.textContent = isUnknown ? '⚠ UNKNOWN' : (isFake ? '🚨 FAKE DETECTED' : '✅ REAL');
                    resultConfidence.textContent = 'Confidence: ' + data.confidence.toFixed(2) + '%';
                    resultDetails.textContent = 'Analyzed ' + data.frames_analyzed + ' face frame(s) from the video';
                    resultNote.textContent = data.frames_analyzed === 0
                        ? 'No faces were detected in any sampled frame — unable to give a reliable result.'
                        : '';
                    confidenceBar.style.width = data.confidence.toFixed(1) + '%';
                    confidenceBar.style.background = isFake
                        ? 'linear-gradient(90deg,#ef4444,#dc2626)'
                        : isUnknown
                            ? 'linear-gradient(90deg,#eab308,#ca8a04)'
                            : 'linear-gradient(90deg,#22c55e,#16a34a)';
                    resultContainer.className = 'result-container ' + (isFake ? 'result-fake' : isUnknown ? 'result-unknown' : 'result-real');
                    resultContainer.style.display = 'block';
                } else {
                    alert('Error: ' + data.message);
                    analyzeBtn.disabled = false;
                }
            } catch (error) {
                loader.style.display = 'none';
                alert('Error: ' + error.message);
                analyzeBtn.disabled = false;
            }
        });

        function resetAnalysis() {
            selectedFile = null; fileInput.value = '';
            previewContainer.style.display = 'none';
            uploadBox.style.display = 'block';
            resultContainer.style.display = 'none';
            analyzeBtn.disabled = false;
            videoSource.src = '';
        }
    </script>
</body>
</html>
"""

# ---------------- ROUTES ----------------

@app.route('/')
def landing():
    return render_template_string(LANDING_PAGE_HTML)

@app.route('/detect')
def home():
    return render_template_string(HOME_HTML)

@app.route('/image')
def image_page():
    return render_template_string(IMAGE_HTML)

@app.route('/video')
def video_page():
    return render_template_string(VIDEO_HTML)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    if not allowed_file(file.filename, 'image'):
        return jsonify({'success': False, 'message': 'Invalid file type. Use PNG, JPG, or JPEG.'})
    try:
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Detect whether face was found (for UI warning)
        frame = cv2.imread(filepath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        no_face = len(faces) == 0

        result, confidence = predict_image(filepath)
        return jsonify({
            'success': True,
            'result': result,
            'confidence': float(confidence),
            'no_face': bool(no_face),
            'filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/analyze_video', methods=['POST'])
def analyze_video_route():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    if not allowed_file(file.filename, 'video'):
        return jsonify({'success': False, 'message': 'Invalid file type. Use MP4, AVI, MOV, or MKV.'})
    try:
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result, confidence, predictions = predict_video(filepath)

        if result is None:
            return jsonify({'success': False, 'message': 'Could not open video file.'})

        return jsonify({
            'success': True,
            'result': result,
            'confidence': float(confidence),
            'frames_analyzed': len(predictions),
            'filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ---------------- RUN ----------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)