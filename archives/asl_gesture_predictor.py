# ──────────────────────────────────────────────────────────────
#  ASL Sign + Gesture Predictor 🤟
#  Compatible with MediaPipe 0.9.x AND 0.10.x
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
#  Handles both mp 0.9.x (mp.solutions) and 0.10.x (submodule import)
# ─────────────────────────────────────────────
try:
    # Try old API first (0.9.x)
    mp_hands          = mp.solutions.hands
    mp_pose           = mp.solutions.pose
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except AttributeError:
    # Fall back to direct submodule imports (0.10.x)
    import mediapipe.python.solutions.hands          as mp_hands
    import mediapipe.python.solutions.pose           as mp_pose
    import mediapipe.python.solutions.drawing_utils  as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
MODEL_PATH   = "asl_effnet_final.h5"
IMG_SIZE     = (160, 160)
CLASS_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DEL','NOTHING','SPACE'
]

# ─────────────────────────────────────────────
#  Hand Gesture Classification
# ─────────────────────────────────────────────
def finger_states(hand_landmarks, handedness="Right"):
    lm = hand_landmarks.landmark
    fingers = []
    # Thumb
    if handedness == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    # Other 4 fingers
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

# ─────────────────────────────────────────────
#  Body Pose Classification
# ─────────────────────────────────────────────
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

    if both_up:
        return "🙌 Both Arms Raised", "Both Arms Raised"
    if l_up and not r_up:
        return "🖐️ Left Arm Raised", "Left Arm Raised"
    if r_up and not l_up:
        return "🖐️ Right Arm Raised", "Right Arm Raised"
    if l_side and r_side:
        return "🤸 T-Pose", "T-Pose"
    if w_close and y(15) < y(0):
        return "🙏 Hands Together Above Head", "Hands Together Above Head"
    if w_close:
        return "🙏 Hands Together", "Hands Together"
    if both_down:
        return "🧍 Rest Position", "Rest Position"
    return None, None

# ─────────────────────────────────────────────
#  Streamlit Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="🤟 ASL + Gesture Predictor", layout="centered")
st.title("🤟 ASL Sign + Gesture Predictor")

MODE = st.sidebar.radio("Choose input mode", ["Image", "Video", "Webcam"])
st.sidebar.markdown("### 🎛️ Feature Toggles")
enable_tts     = st.sidebar.toggle("🔊 Text-to-Speech",         value=True)
show_asl       = st.sidebar.toggle("🔡 ASL Letter Detection",   value=True)
show_hand      = st.sidebar.toggle("🖐️ Hand Gesture Detection", value=True)
show_pose      = st.sidebar.toggle("🧍 Body Pose Detection",    value=True)
show_landmarks = st.sidebar.toggle("📍 Show Landmarks",         value=True)
st.sidebar.markdown("---")
st.sidebar.info("💡 **EfficientNet** handles ASL letters.\n\n**MediaPipe** handles hand + body gestures.")

# ─────────────────────────────────────────────
#  Load ASL Model
# ─────────────────────────────────────────────
@st.cache_resource
def get_model(path):
    return load_model(path)

model = get_model(MODEL_PATH)
st.success("✅ ASL Model ready!")

# ─────────────────────────────────────────────
#  MediaPipe Instances (cached for image/video mode)
# ─────────────────────────────────────────────
@st.cache_resource
def get_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

@st.cache_resource
def get_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

# ─────────────────────────────────────────────
#  Core Helper Functions
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

    hand_gestures = []   # list of (display_str, clean_name)
    if hand_res and hand_res.multi_hand_landmarks:
        for hand_lm, hand_info in zip(
            hand_res.multi_hand_landmarks,
            hand_res.multi_handedness
        ):
            if draw:
                mp_drawing.draw_landmarks(
                    out, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            h_label       = hand_info.classification[0].label
            display, clean = classify_hand_gesture(hand_lm, h_label)
            hand_gestures.append((f"{h_label}: {display}", clean))

    pose_display, pose_clean = None, None
    if pose_res and pose_res.pose_landmarks:
        if draw:
            mp_drawing.draw_landmarks(
                out, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
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
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 8 + (i+1)*line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,120), 2)
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
    st.header("🖼️ Predict from Image")
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        asl_label, asl_conf = None, None
        if show_asl:
            asl_label, asl_conf = predict_sign(img_bgr)

        annotated, hand_gestures, pose_display, pose_clean = run_mediapipe(
            img_bgr, get_hands(), get_pose(), draw=show_landmarks
        )
        annotated = draw_overlay(annotated, asl_label, asl_conf, hand_gestures, pose_display)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if asl_label and show_asl:
                st.metric("ASL Letter", asl_label, f"{asl_conf*100:.1f}%")
        with col2:
            if hand_gestures and show_hand:
                st.metric("Hand Gesture", hand_gestures[0][0].split(": ",1)[-1])
                if len(hand_gestures) > 1:
                    st.metric("Hand 2", hand_gestures[1][0].split(": ",1)[-1])
        with col3:
            if pose_display and show_pose:
                st.metric("Body Pose", pose_display)

        if enable_tts:
            parts = []
            if asl_label and show_asl and asl_label not in ("NOTHING","SPACE","DEL"):
                parts.append(f"ASL letter {asl_label}")
            if hand_gestures and show_hand:
                parts.append(hand_gestures[0][1])
            if pose_clean and show_pose:
                parts.append(pose_clean)
            if parts:
                speak_text(". ".join(parts))

# ─────────────────────────────────────────────
#  VIDEO MODE
# ─────────────────────────────────────────────
elif MODE == "Video":
    st.header("🎞️ Predict from Video")
    uploaded = st.file_uploader("Upload video", type=["mp4","mov","avi"])

    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()

        cap          = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count  = 0
        frame_preds, frame_hands, frame_poses = [], [], []

        stframe  = st.empty()
        progress = st.progress(0)

        # Use static_image_mode=True for video to avoid tracking state issues
        v_hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                                  min_detection_confidence=0.6)
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
                frame, v_hands, v_pose, draw=show_landmarks
            )
            if hand_gestures: frame_hands.append(hand_gestures[0][1])
            if pose_clean:    frame_poses.append(pose_clean)

            annotated = draw_overlay(annotated, asl_label, asl_conf, hand_gestures, pose_display)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            frame_count += 1
            if total_frames > 0:
                progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        v_hands.close()
        v_pose.close()
        progress.empty()

        st.markdown("### 📊 Video Summary")
        col1, col2, col3 = st.columns(3)
        final_asl = final_hand = final_pose = None

        with col1:
            if frame_preds and show_asl:
                final_asl = max(set(frame_preds), key=frame_preds.count)
                st.metric("Most Common ASL", final_asl)
        with col2:
            if frame_hands and show_hand:
                final_hand = max(set(frame_hands), key=frame_hands.count)
                st.metric("Most Common Hand Gesture", final_hand)
        with col3:
            if frame_poses and show_pose:
                final_pose = max(set(frame_poses), key=frame_poses.count)
                st.metric("Most Common Pose", final_pose)

        if enable_tts:
            parts = []
            if final_asl:  parts.append(f"ASL letter {final_asl}")
            if final_hand: parts.append(final_hand)
            if final_pose: parts.append(final_pose)
            if parts:
                speak_text(". ".join(parts))

# ─────────────────────────────────────────────
#  WEBCAM MODE
# ─────────────────────────────────────────────
else:
    st.header("📷 Live Webcam — ASL + Gesture")

    for key, default in [
        ("sentence",      ""),
        ("stable_letter", ""),
        ("stable_count",  0),
        ("last_added",    ""),
        ("last_spoken",   ""),
        ("last_hand_g",   ""),
        ("last_pose_g",   ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    pred_box     = st.empty()
    hand_box     = st.empty()
    pose_box     = st.empty()
    sentence_box = st.empty()
    audio_box    = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Sentence"):
            for k in ("sentence","last_added","last_spoken","last_hand_g","last_pose_g"):
                st.session_state[k] = ""
            audio_box.empty()
    with col2:
        if st.button("🔊 Speak Full Sentence"):
            if st.session_state.sentence.strip() and enable_tts:
                speak_text(st.session_state.sentence.strip(), audio_box)

    class SignGestureProcessor(VideoProcessorBase):
        def __init__(self):
            self.asl_label     = "..."
            self.asl_conf      = 0.0
            self.hand_gestures = []
            self.pose_display  = None
            self.pose_clean    = None
            # Own MP instances per processor to avoid thread conflicts
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )

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
                for hand_lm, hand_info in zip(
                    hand_res.multi_hand_landmarks,
                    hand_res.multi_handedness
                ):
                    if show_landmarks:
                        mp_drawing.draw_landmarks(
                            img, hand_lm, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    h_label        = hand_info.classification[0].label
                    display, clean = classify_hand_gesture(hand_lm, h_label)
                    self.hand_gestures.append((f"{h_label}: {display}", clean))

            self.pose_display, self.pose_clean = None, None
            if pose_res and pose_res.pose_landmarks:
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        img, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                self.pose_display, self.pose_clean = classify_pose_gesture(pose_res.pose_landmarks)

            img = draw_overlay(
                img,
                self.asl_label if show_asl else None,
                self.asl_conf,
                self.hand_gestures if show_hand else None,
                self.pose_display  if show_pose else None
            )
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

            if show_asl:
                pred_box.markdown(f"## 🔡 ASL: `{vp.asl_label}` — {vp.asl_conf*100:.1f}%")
            if show_hand:
                hand_box.markdown(
                    "### 🖐️ " + "  |  ".join(g[0] for g in vp.hand_gestures)
                    if vp.hand_gestures else "### 🖐️ No hand detected"
                )
            if show_pose:
                pose_box.markdown(f"### 🧍 Pose: `{vp.pose_display or 'None'}`")

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

                sentence_box.markdown(f"### ✍️ Sentence: `{st.session_state.sentence}`")

            # Speak new hand gesture via TTS
            if show_hand and enable_tts and vp.hand_gestures:
                clean = vp.hand_gestures[0][1]
                if clean != st.session_state.last_hand_g and clean != "Unknown Gesture":
                    speak_text(clean, audio_box)
                    st.session_state.last_hand_g = clean

            # Speak new pose gesture via TTS
            if show_pose and enable_tts and vp.pose_clean:
                if vp.pose_clean != st.session_state.last_pose_g:
                    speak_text(vp.pose_clean, audio_box)
                    st.session_state.last_pose_g = vp.pose_clean

        time.sleep(0.2)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with **Streamlit** • **EfficientNet** ASL Model • **MediaPipe** Hands + Pose")
