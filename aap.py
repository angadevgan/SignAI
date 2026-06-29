# ──────────────────────────────────────────────────────────────
#  ASL Sign Predictor – Streamlit Interface (Image • Video • Cam)
# ──────────────────────────────────────────────────────────────
#
#  • Upload **image**  (.jpg/.png)  → predicts sign
#  • Upload **video**  (.mp4/.mov)  → per‑frame overlay + majority vote
#  • Use **webcam** (WebRTC)        → live, continuous prediction
#
#  Put your trained model (e.g. asl_effnet_final.h5) beside this file.
#  Requires: streamlit>=1.25, streamlit‑webrtc, opencv‑python, av, tensorflow
# ──────────────────────────────────────────────────────────────

import time, tempfile, cv2, numpy as np, av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------- Configuration -------------------------------------
MODEL_PATH   = "asl_effnet_final.h5"          # change if needed
IMG_SIZE     = (160, 160)
CLASS_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DEL','NOTHING','SPACE'
]

# ---------- Streamlit page setup ------------------------------
st.set_page_config(page_title="🤟 ASL Sign Predictor", layout="centered")
st.title("🤟 ASL Sign Predictor")

# ---------- Custom CSS ----------------------------------------
st.markdown(
    """
    <style>
    .big-pred {font-size:3.5rem;font-weight:900;color:#0984e3;text-align:center;margin-top:0.3em;}
    .confidence {font-size:1.25rem;color:#636e72;text-align:center;margin-bottom:1.2rem;}
    .footer {font-size:0.9rem;color:#b2bec3;text-align:center;margin-top:3em;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model only once (cached) ---------------------
@st.cache_resource(show_spinner="Loading model …", max_entries=1)
def get_model(path: str):
    """Load and cache the Keras model so it persists across reruns."""
    return load_model(path)

try:
    model = get_model(MODEL_PATH)
    st.success("✅ Model ready!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---------- Helper functions ----------------------------------
def prepare_img(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, IMG_SIZE)
    tensor  = preprocess_input(img_res.astype(np.float32))
    return np.expand_dims(tensor, 0)

def predict_sign(img_bgr: np.ndarray):
    x      = prepare_img(img_bgr)
    probs  = model.predict(x, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx])

# ---------- Mode selector -------------------------------------
MODE = st.sidebar.radio("Choose input mode", ["Image", "Video", "Webcam"], index=0)

# ---------- IMAGE MODE ----------------------------------------
if MODE == "Image":
    st.header("🖼️ Predict from Image")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label, conf = predict_sign(img_bgr)

        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.markdown(f"<div class='big-pred'>{label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence'>Confidence: {conf*100:.2f}%</div>",
                    unsafe_allow_html=True)

# ---------- VIDEO MODE ----------------------------------------
elif MODE == "Video":
    st.header("🎞️ Predict from Video")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        frame_preds, current = [], 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        stframe  = st.empty()
        progress = st.progress(0.0, "Processing video …")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current += 1
            label, _ = predict_sign(frame)
            frame_preds.append(label)

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 255, 0), 2)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            progress.progress(min(current / total_frames, 1.0))
        cap.release()

        if frame_preds:
            final = max(set(frame_preds), key=frame_preds.count)
            st.markdown(f"<div class='big-pred'>{final}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence'>(majority vote across {len(frame_preds)} frames)</div>",
                        unsafe_allow_html=True)

# ---------- WEBCAM MODE ---------------------------------------
else:
    st.header("📷 Live Webcam Prediction")
    st.info("Allow camera access and perform a sign in view.")

    # Real‑time label placeholder
    pred_box = st.empty()

    class SignProcessor(VideoProcessorBase):
        def __init__(self):
            self.label_txt = "…"
        def recv(self, frame: av.VideoFrame):
            img = frame.to_ndarray(format="bgr24")
            label, conf = predict_sign(img)
            self.label_txt = f"{label} ({conf*100:.1f}%)"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 0, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="asl-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=SignProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Safely poll the processor’s label every 0.2 s while the stream is playing
    while ctx and ctx.state.playing:
        if ctx.video_processor:
            pred_box.markdown(f"<div class='big-pred'>{ctx.video_processor.label_txt}</div>",
                              unsafe_allow_html=True)
        time.sleep(0.2)

# ---------- Footer --------------------------------------------
st.markdown(
    "<div class='footer'>Built with <b>Streamlit</b> • Model: EfficientNet‑B0 fine‑tuned on ASL Alphabet dataset</div>",
    unsafe_allow_html=True,
)
