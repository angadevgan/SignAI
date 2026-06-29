# ──────────────────────────────────────────────────────────────
#  ASL Sign + Gesture Predictor 🤟
#  Compatible with MediaPipe 0.9.x AND 0.10.x
#  Enhanced UI — Dark Cyberpunk Theme
# ──────────────────────────────────────────────────────────────

import time, tempfile, cv2, numpy as np, av, io
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from gtts import gTTS
import mediapipe as mp

# ─────────────────────────────────────────────
#  MediaPipe Compatibility Layer
# ─────────────────────────────────────────────
try:
    mp_hands          = mp.solutions.hands
    mp_pose           = mp.solutions.pose
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except AttributeError:
    import mediapipe.python.solutions.hands          as mp_hands
    import mediapipe.python.solutions.pose           as mp_pose
    import mediapipe.python.solutions.drawing_utils  as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
MODEL_PATH   = "models/asl_effnet_final.h5"
IMG_SIZE     = (160, 160)
CLASS_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DEL','NOTHING','SPACE'
]

# ─────────────────────────────────────────────
#  Page Config (MUST be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SignAI — ASL + Gesture Predictor",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Custom CSS — Dark Cyberpunk Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #040810 !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Animated grid background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,200,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 20%, rgba(0,255,180,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(100,0,255,0.08) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: drift 20s ease-in-out infinite alternate;
}

@keyframes drift {
    0%   { transform: translate(0, 0); }
    100% { transform: translate(30px, -20px); }
}

/* Main content above bg */
[data-testid="stMain"], .main .block-container {
    position: relative;
    z-index: 1;
    padding: 1.5rem 2rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #040810 100%) !important;
    border-right: 1px solid rgba(0,255,180,0.15) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00ffb4, #7c3aed, #00ffb4);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
    0%   { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

[data-testid="stSidebarContent"] {
    padding: 1.5rem 1rem !important;
}

/* Sidebar radio */
[data-testid="stSidebar"] .stRadio > label {
    color: #94a3b8 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

/* Radio buttons */
[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
    background: rgba(0,255,180,0.2) !important;
    border-color: #00ffb4 !important;
}

/* Toggle switches */
[data-testid="stSidebar"] [data-testid="stToggle"] > div > div {
    background: #1e293b !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] input:checked ~ div {
    background: linear-gradient(90deg, #00ffb4, #06b6d4) !important;
}

/* Toggle labels */
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: #cbd5e1 !important;
    font-size: 0.88rem !important;
}

/* ── Sidebar Info Box ── */
[data-testid="stSidebar"] .stAlert {
    background: rgba(0,255,180,0.06) !important;
    border: 1px solid rgba(0,255,180,0.2) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

/* ── Headers ── */
h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 900 !important;
    font-size: 2.2rem !important;
    background: linear-gradient(135deg, #00ffb4 0%, #06b6d4 50%, #7c3aed 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 0.25rem !important;
    filter: drop-shadow(0 0 20px rgba(0,255,180,0.3));
}

h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    letter-spacing: 0.04em !important;
}

h2 { font-size: 1.3rem !important; }
h3 { font-size: 1.05rem !important; color: #94a3b8 !important; }

/* Markdown text */
p, .stMarkdown p { color: #94a3b8 !important; }

/* ── Success / Alert Boxes ── */
[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(0,255,180,0.07) !important;
    border: 1px solid rgba(0,255,180,0.3) !important;
    border-radius: 12px !important;
    color: #00ffb4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,180,0.1), rgba(124,58,237,0.1)) !important;
    border: 1px solid rgba(0,255,180,0.3) !important;
    border-radius: 10px !important;
    color: #00ffb4 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,255,180,0.25), rgba(124,58,237,0.2)) !important;
    border-color: #00ffb4 !important;
    box-shadow: 0 0 20px rgba(0,255,180,0.3), 0 0 40px rgba(0,255,180,0.1) !important;
    transform: translateY(-1px) !important;
    color: #ffffff !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(0,255,180,0.2) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,255,180,0.5) !important;
    background: rgba(0,255,180,0.03) !important;
}

[data-testid="stFileUploader"] label {
    color: #64748b !important;
    font-size: 0.85rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(0,255,180,0.04)) !important;
    border: 1px solid rgba(0,255,180,0.15) !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ffb4, transparent);
    opacity: 0.6;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(0,255,180,0.4) !important;
    box-shadow: 0 4px 20px rgba(0,255,180,0.1) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #00ffb4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    color: #7c3aed !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Progress Bar ── */
[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00ffb4, #7c3aed) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(0,255,180,0.4) !important;
    transition: width 0.3s ease !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(0,255,180,0.15) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,255,180,0.05) !important;
}

/* ── Audio Player ── */
audio {
    width: 100% !important;
    border-radius: 10px !important;
    filter: invert(1) hue-rotate(150deg) !important;
    opacity: 0.85 !important;
}

/* ── Markdown info boxes ── */
.stMarkdown h2, .stMarkdown h3 {
    border-bottom: 1px solid rgba(0,255,180,0.15) !important;
    padding-bottom: 0.4rem !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,255,180,0.12) !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #040810; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00ffb4, #7c3aed);
    border-radius: 3px;
}

/* ── WebRTC video container ── */
[data-testid="stCustomComponentV1"] iframe {
    border-radius: 16px !important;
    border: 1px solid rgba(0,255,180,0.2) !important;
    box-shadow: 0 0 40px rgba(0,255,180,0.08) !important;
}

/* ── Sentence display ── */
.sentence-display {
    background: linear-gradient(135deg, rgba(0,255,180,0.05), rgba(124,58,237,0.05));
    border: 1px solid rgba(0,255,180,0.2);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #e2e8f0;
    letter-spacing: 0.05em;
    min-height: 56px;
    display: flex;
    align-items: center;
    margin: 0.5rem 0;
    position: relative;
    overflow: hidden;
}

.sentence-display::after {
    content: '▋';
    color: #00ffb4;
    animation: blink 1s step-end infinite;
    margin-left: 2px;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,255,180,0.08);
    border: 1px solid rgba(0,255,180,0.25);
    border-radius: 999px;
    padding: 4px 14px;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #00ffb4;
    text-transform: uppercase;
}

.status-badge .dot {
    width: 6px; height: 6px;
    background: #00ffb4;
    border-radius: 50%;
    box-shadow: 0 0 6px #00ffb4;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

/* ── ASL detection box ── */
.asl-box {
    background: linear-gradient(135deg, rgba(0,255,180,0.06), rgba(6,182,212,0.06));
    border: 1px solid rgba(0,255,180,0.2);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    color: #00ffb4;
    font-size: 1rem;
    margin: 0.3rem 0;
}

/* ── Hand/pose detection box ── */
.detect-box {
    background: rgba(124,58,237,0.06);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 14px;
    padding: 0.8rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    color: #a78bfa;
    font-size: 0.9rem;
    margin: 0.3rem 0;
}

/* Mode header tag */
.mode-tag {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,255,180,0.12), rgba(124,58,237,0.12));
    border: 1px solid rgba(0,255,180,0.25);
    border-radius: 8px;
    padding: 3px 12px;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #00ffb4;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Sidebar section label */
.sidebar-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #475569;
    text-transform: uppercase;
    margin: 1rem 0 0.4rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.2rem;">
    <div>
        <h1>SignAI</h1>
        <p style="color:#475569; font-family:'DM Sans',sans-serif; font-size:0.95rem; margin-top:0.1rem;">
            ASL Letter Recognition + Hand & Body Gesture Detection
        </p>
    </div>
    <div class="status-badge">
        <span class="dot"></span>
        System Online
    </div>
</div>
<hr/>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h1 style="font-size:1.1rem !important; margin-bottom:1rem;">🤟 SignAI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Input Mode</div>', unsafe_allow_html=True)
    MODE = st.radio("", ["Image", "Video", "Webcam"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">Feature Toggles</div>', unsafe_allow_html=True)
    enable_tts     = st.toggle("🔊 Text-to-Speech",         value=True)
    show_asl       = st.toggle("🔡 ASL Letter Detection",   value=True)
    show_hand      = st.toggle("🖐️ Hand Gesture Detection", value=True)
    show_pose      = st.toggle("🧍 Body Pose Detection",    value=True)
    show_landmarks = st.toggle("📍 Show Landmarks",         value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; line-height:1.7;">
        <span style="color:#00ffb4;">●</span> <b style="color:#64748b;">EfficientNet</b> → ASL letters<br>
        <span style="color:#a78bfa;">●</span> <b style="color:#64748b;">MediaPipe</b> → Hand + body<br>
        <span style="color:#06b6d4;">●</span> <b style="color:#64748b;">gTTS</b> → Speech output
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Gesture Classification Logic
# ─────────────────────────────────────────────
def finger_states(hand_landmarks, handedness="Right"):
    lm = hand_landmarks.landmark
    fingers = []
    if handedness == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)
    return fingers

def classify_hand_gesture(hand_landmarks, handedness="Right"):
    f = finger_states(hand_landmarks, handedness)
    gesture_map = {
        (0,0,0,0,0): ("✊", "Fist"),
        (1,1,1,1,1): ("🖐️", "Open Hand"),
        (0,1,0,0,0): ("☝️", "Pointing Up"),
        (0,1,1,0,0): ("✌️", "Peace"),
        (1,1,0,0,1): ("🤟", "I Love You"),
        (1,0,0,0,0): ("👍", "Thumbs Up"),
        (0,0,0,0,1): ("🤙", "Call Me"),
        (0,1,1,1,1): ("✋", "Four Fingers"),
        (1,1,1,0,0): ("🤌", "Three Fingers"),
        (0,1,0,0,1): ("🤘", "Rock On"),
    }
    emoji, name = gesture_map.get(tuple(f), ("🤚", "Unknown Gesture"))
    return f"{emoji} {name}", name

def classify_pose_gesture(pose_landmarks):
    lm = pose_landmarks.landmark
    def y(i): return lm[i].y
    def x(i): return lm[i].x
    def vis(i): return lm[i].visibility
    if vis(15) < 0.5 and vis(16) < 0.5:
        return None, None
    both_up   = y(15) < y(11) and y(16) < y(12)
    both_down = y(15) > y(23) and y(16) > y(24)
    l_up      = y(15) < y(11)
    r_up      = y(16) < y(12)
    l_side    = abs(y(15) - y(11)) < 0.15
    r_side    = abs(y(16) - y(12)) < 0.15
    w_close   = abs(x(15) - x(16)) < 0.1 and abs(y(15) - y(16)) < 0.1
    if both_up:   return "🙌 Both Arms Raised", "Both Arms Raised"
    if l_up and not r_up: return "🖐️ Left Arm Raised", "Left Arm Raised"
    if r_up and not l_up: return "🖐️ Right Arm Raised", "Right Arm Raised"
    if l_side and r_side: return "🤸 T-Pose", "T-Pose"
    if w_close and y(15) < y(0): return "🙏 Hands Together Above Head", "Hands Together Above Head"
    if w_close:   return "🙏 Hands Together", "Hands Together"
    if both_down: return "🧍 Rest Position", "Rest Position"
    return None, None

# ─────────────────────────────────────────────
#  Load ASL Model
# ─────────────────────────────────────────────
@st.cache_resource
def get_model(path):
    return load_model(path)

model = get_model(MODEL_PATH)
st.markdown('<div class="status-badge" style="margin-bottom:1rem;"><span class="dot"></span>EfficientNet Model Loaded</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MediaPipe Instances
# ─────────────────────────────────────────────
@st.cache_resource
def get_hands():
    return mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=0.6, min_tracking_confidence=0.5)

@st.cache_resource
def get_pose():
    return mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5)

# ─────────────────────────────────────────────
#  Core Functions
# ─────────────────────────────────────────────
def prepare_img(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, IMG_SIZE)
    tensor  = preprocess_input(img_res.astype(np.float32))
    return np.expand_dims(tensor, 0)

def predict_sign(img_bgr):
    x     = prepare_img(img_bgr)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx])

def run_mediapipe(img_bgr, hands_obj, pose_obj, draw=True):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    hand_res = hands_obj.process(img_rgb) if show_hand else None
    pose_res = pose_obj.process(img_rgb)  if show_pose else None
    img_rgb.flags.writeable = True
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hand_gestures = []
    if hand_res and hand_res.multi_hand_landmarks:
        for hand_lm, hand_info in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
            if draw:
                mp_drawing.draw_landmarks(out, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            h_label = hand_info.classification[0].label
            display, clean = classify_hand_gesture(hand_lm, h_label)
            hand_gestures.append((f"{h_label}: {display}", clean))
    pose_display, pose_clean = None, None
    if pose_res and pose_res.pose_landmarks:
        if draw:
            mp_drawing.draw_landmarks(out, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        pose_display, pose_clean = classify_pose_gesture(pose_res.pose_landmarks)
    return out, hand_gestures, pose_display, pose_clean

def draw_overlay(img, asl_label=None, asl_conf=None, hand_gestures=None, pose_display=None):
    lines = []
    if asl_label and show_asl:
        lines.append(f"ASL: {asl_label}  ({asl_conf*100:.1f}%)")
    if hand_gestures and show_hand:
        for g, _ in hand_gestures:
            lines.append(g)
    if pose_display and show_pose:
        lines.append(f"Pose: {pose_display}")
    if not lines:
        return img
    overlay = img.copy()
    line_h  = 28
    box_h   = len(lines) * line_h + 16
    h, w    = img.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, box_h), (4, 8, 16), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 8 + (i+1)*line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,180), 2)
    return img

def generate_audio_bytes(text):
    tts = gTTS(text=text, lang='en')
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf

def speak_text(text, placeholder=None):
    if not text or text.strip() in ("", "..."):
        return
    try:
        audio = generate_audio_bytes(text)
        (placeholder or st).audio(audio, format="audio/mp3")
    except Exception as e:
        st.warning(f"TTS error: {e}")

# ─────────────────────────────────────────────
#  IMAGE MODE
# ─────────────────────────────────────────────
if MODE == "Image":
    st.markdown('<div class="mode-tag">📸 Image Mode</div>', unsafe_allow_html=True)
    st.markdown("### Upload an Image")

    uploaded = st.file_uploader("Drop your image here", type=["jpg","jpeg","png"],
                                 label_visibility="collapsed")

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        asl_label, asl_conf = None, None
        if show_asl:
            asl_label, asl_conf = predict_sign(img_bgr)

        annotated, hand_gestures, pose_display, pose_clean = run_mediapipe(
            img_bgr, get_hands(), get_pose(), draw=show_landmarks)
        annotated = draw_overlay(annotated, asl_label, asl_conf, hand_gestures, pose_display)

        col_img, col_res = st.columns([3, 2], gap="large")

        with col_img:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_res:
            st.markdown("#### Detection Results")

            if asl_label and show_asl:
                st.markdown(f"""
                <div class="asl-box">
                    <div style="font-size:0.65rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">ASL Letter</div>
                    <div style="font-size:2rem;font-weight:700;">{asl_label}</div>
                    <div style="font-size:0.8rem;color:#475569;">Confidence: {asl_conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            if hand_gestures and show_hand:
                for i, (disp, clean) in enumerate(hand_gestures):
                    st.markdown(f"""
                    <div class="detect-box">
                        <div style="font-size:0.6rem;color:#7c3aed;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px;">Hand {i+1}</div>
                        <div>{disp}</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif show_hand:
                st.markdown('<div class="detect-box" style="color:#475569;">🖐️ No hand detected</div>', unsafe_allow_html=True)

            if pose_display and show_pose:
                st.markdown(f"""
                <div class="detect-box">
                    <div style="font-size:0.6rem;color:#7c3aed;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px;">Body Pose</div>
                    <div>{pose_display}</div>
                </div>
                """, unsafe_allow_html=True)

            if enable_tts:
                parts = []
                if asl_label and show_asl and asl_label not in ("NOTHING","SPACE","DEL"):
                    parts.append(f"ASL letter {asl_label}")
                if hand_gestures and show_hand:
                    parts.append(hand_gestures[0][1])
                if pose_clean and show_pose:
                    parts.append(pose_clean)
                if parts:
                    st.markdown("---")
                    st.markdown('<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">🔊 Audio Output</div>', unsafe_allow_html=True)
                    speak_text(". ".join(parts))

    else:
        st.markdown("""
        <div style="border:2px dashed rgba(0,255,180,0.15);border-radius:16px;padding:3rem;text-align:center;color:#334155;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">🖼️</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;">
                Upload an image to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  VIDEO MODE
# ─────────────────────────────────────────────
elif MODE == "Video":
    st.markdown('<div class="mode-tag">🎞️ Video Mode</div>', unsafe_allow_html=True)
    st.markdown("### Upload a Video")

    uploaded = st.file_uploader("Drop your video here", type=["mp4","mov","avi"],
                                 label_visibility="collapsed")

    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()

        cap          = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count  = 0
        frame_preds, frame_hands, frame_poses = [], [], []

        st.markdown("**Processing frames...**")
        stframe  = st.empty()
        progress = st.progress(0)
        stat_box = st.empty()

        v_hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)
        v_pose  = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.6)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            asl_label, asl_conf = None, None
            if show_asl:
                asl_label, asl_conf = predict_sign(frame)
                frame_preds.append(asl_label)

            annotated, hand_gestures, pose_display, pose_clean = run_mediapipe(
                frame, v_hands, v_pose, draw=show_landmarks)
            if hand_gestures: frame_hands.append(hand_gestures[0][1])
            if pose_clean:    frame_poses.append(pose_clean)

            annotated = draw_overlay(annotated, asl_label, asl_conf, hand_gestures, pose_display)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            frame_count += 1
            pct = min(frame_count / total_frames, 1.0) if total_frames > 0 else 0
            progress.progress(pct)
            stat_box.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;color:#475569;">Frame {frame_count} / {total_frames} — {pct*100:.0f}%</div>', unsafe_allow_html=True)

        cap.release()
        v_hands.close()
        v_pose.close()
        progress.empty()
        stat_box.empty()

        st.markdown("---")
        st.markdown("### 📊 Video Summary")
        final_asl = final_hand = final_pose = None

        c1, c2, c3 = st.columns(3)
        with c1:
            if frame_preds and show_asl:
                final_asl = max(set(frame_preds), key=frame_preds.count)
                st.metric("Most Common ASL", final_asl)
        with c2:
            if frame_hands and show_hand:
                final_hand = max(set(frame_hands), key=frame_hands.count)
                st.metric("Most Common Gesture", final_hand)
        with c3:
            if frame_poses and show_pose:
                final_pose = max(set(frame_poses), key=frame_poses.count)
                st.metric("Most Common Pose", final_pose)

        if enable_tts:
            parts = []
            if final_asl:  parts.append(f"ASL letter {final_asl}")
            if final_hand: parts.append(final_hand)
            if final_pose: parts.append(final_pose)
            if parts:
                st.markdown('<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin:0.5rem 0 4px;">🔊 Audio Summary</div>', unsafe_allow_html=True)
                speak_text(". ".join(parts))

    else:
        st.markdown("""
        <div style="border:2px dashed rgba(0,255,180,0.15);border-radius:16px;padding:3rem;text-align:center;color:#334155;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">🎞️</div>
            <div style="font-family:'Orbitron',sans-serif;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;">
                Upload a video to begin frame analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  WEBCAM MODE
# ─────────────────────────────────────────────
else:
    st.markdown('<div class="mode-tag">📷 Live Webcam Mode</div>', unsafe_allow_html=True)

    for key, default in [
        ("sentence",""), ("stable_letter",""), ("stable_count",0),
        ("last_added",""), ("last_spoken",""), ("last_hand_g",""), ("last_pose_g",""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Layout: video left, detections right
    col_vid, col_info = st.columns([3, 2], gap="large")

    with col_info:
        st.markdown("#### Live Detections")
        pred_box = st.empty()
        hand_box = st.empty()
        pose_box = st.empty()

        st.markdown("---")
        st.markdown("#### Sentence Builder")
        sentence_box = st.empty()
        audio_box    = st.empty()

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("🗑️ Clear"):
                for k in ("sentence","last_added","last_spoken","last_hand_g","last_pose_g"):
                    st.session_state[k] = ""
                audio_box.empty()
        with btn2:
            if st.button("🔊 Speak"):
                if st.session_state.sentence.strip() and enable_tts:
                    speak_text(st.session_state.sentence.strip(), audio_box)

        st.markdown("""
        <div style="font-size:0.72rem;color:#334155;line-height:1.8;margin-top:0.5rem;">
            <span style="color:#00ffb4;">SPACE</span> → speak + add space<br>
            <span style="color:#00ffb4;">DEL</span> → delete last letter<br>
            <span style="color:#00ffb4;">NOTHING</span> → ignored
        </div>
        """, unsafe_allow_html=True)

    with col_vid:
        class SignGestureProcessor(VideoProcessorBase):
            def __init__(self):
                self.asl_label     = "..."
                self.asl_conf      = 0.0
                self.hand_gestures = []
                self.pose_display  = None
                self.pose_clean    = None
                self._hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                              min_detection_confidence=0.6, min_tracking_confidence=0.5)
                self._pose  = mp_pose.Pose(static_image_mode=False,
                                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                if show_asl:
                    self.asl_label, self.asl_conf = predict_sign(img)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                hand_res = self._hands.process(img_rgb) if show_hand else None
                pose_res = self._pose.process(img_rgb)  if show_pose else None
                img_rgb.flags.writeable = True
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                self.hand_gestures = []
                if hand_res and hand_res.multi_hand_landmarks:
                    for hand_lm, hand_info in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                        if show_landmarks:
                            mp_drawing.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                        h_label = hand_info.classification[0].label
                        display, clean = classify_hand_gesture(hand_lm, h_label)
                        self.hand_gestures.append((f"{h_label}: {display}", clean))
                self.pose_display, self.pose_clean = None, None
                if pose_res and pose_res.pose_landmarks:
                    if show_landmarks:
                        mp_drawing.draw_landmarks(img, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    self.pose_display, self.pose_clean = classify_pose_gesture(pose_res.pose_landmarks)
                img = draw_overlay(img,
                    self.asl_label if show_asl else None, self.asl_conf,
                    self.hand_gestures if show_hand else None,
                    self.pose_display  if show_pose else None)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(
            key="asl-gesture-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SignGestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    STABLE_THRESHOLD = 5

    while ctx and ctx.state.playing:
        if ctx.video_processor:
            vp = ctx.video_processor

            with col_info:
                if show_asl:
                    pred_box.markdown(f"""
                    <div class="asl-box">
                        <div style="font-size:0.6rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">ASL Detected</div>
                        <div style="font-size:1.6rem;font-weight:700;">{vp.asl_label} <span style="font-size:0.8rem;color:#475569;">{vp.asl_conf*100:.1f}%</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                if show_hand:
                    if vp.hand_gestures:
                        hand_text = "  |  ".join(g[0] for g in vp.hand_gestures)
                        hand_box.markdown(f'<div class="detect-box">🖐️ {hand_text}</div>', unsafe_allow_html=True)
                    else:
                        hand_box.markdown('<div class="detect-box" style="color:#334155;">🖐️ No hand detected</div>', unsafe_allow_html=True)

                if show_pose:
                    pose_box.markdown(
                        f'<div class="detect-box">🧍 {vp.pose_display}</div>' if vp.pose_display
                        else '<div class="detect-box" style="color:#334155;">🧍 No pose detected</div>',
                        unsafe_allow_html=True)

            # ASL sentence building
            if show_asl:
                cur = vp.asl_label
                if cur == st.session_state.stable_letter:
                    st.session_state.stable_count += 1
                else:
                    st.session_state.stable_letter = cur
                    st.session_state.stable_count  = 1

                if st.session_state.stable_count == STABLE_THRESHOLD:
                    if cur == "SPACE":
                        words = st.session_state.sentence.strip().split()
                        if words and enable_tts:
                            w = words[-1]
                            if w != st.session_state.last_spoken:
                                speak_text(w, audio_box)
                                st.session_state.last_spoken = w
                        st.session_state.sentence  += " "
                        st.session_state.last_added = ""
                    elif cur == "DEL":
                        st.session_state.sentence   = st.session_state.sentence[:-1]
                        st.session_state.last_added = ""
                    elif cur not in ("NOTHING","..."):
                        if cur != st.session_state.last_added:
                            st.session_state.sentence  += cur
                            st.session_state.last_added = cur

                sentence_box.markdown(
                    f'<div class="sentence-display">{st.session_state.sentence or "&nbsp;"}</div>',
                    unsafe_allow_html=True)

            if show_hand and enable_tts and vp.hand_gestures:
                clean = vp.hand_gestures[0][1]
                if clean != st.session_state.last_hand_g and clean != "Unknown Gesture":
                    speak_text(clean, audio_box)
                    st.session_state.last_hand_g = clean

            if show_pose and enable_tts and vp.pose_clean:
                if vp.pose_clean != st.session_state.last_pose_g:
                    speak_text(vp.pose_clean, audio_box)
                    st.session_state.last_pose_g = vp.pose_clean

        time.sleep(0.2)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;color:#1e293b;font-size:0.75rem;">
    <div style="font-family:'Orbitron',sans-serif;letter-spacing:0.1em;">SignAI © 2024</div>
    <div>EfficientNet · MediaPipe · Streamlit · gTTS</div>
</div>
""", unsafe_allow_html=True)