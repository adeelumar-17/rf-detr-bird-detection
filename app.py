import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import time
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroWatch — Bird Detection",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

:root {
    --bg:        #0a0e17;
    --surface:   #111827;
    --border:    #1f2d3d;
    --accent:    #00e5ff;
    --warn:      #ffb300;
    --danger:    #ff1744;
    --safe:      #00e676;
    --text:      #e0e6ed;
    --muted:     #4a5568;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 { font-family: var(--sans) !important; font-weight: 700 !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    letter-spacing: 1px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

.stSlider > div > div > div { background: var(--accent) !important; }

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* Risk badge */
.risk-safe    { background:#003d1f; border:1px solid var(--safe);   color:var(--safe);   padding:6px 18px; border-radius:4px; font-family:var(--mono); font-size:1.1rem; display:inline-block; }
.risk-low     { background:#3d2a00; border:1px solid var(--warn);   color:var(--warn);   padding:6px 18px; border-radius:4px; font-family:var(--mono); font-size:1.1rem; display:inline-block; }
.risk-high    { background:#3d0010; border:1px solid var(--danger);  color:var(--danger); padding:6px 18px; border-radius:4px; font-family:var(--mono); font-size:1.1rem; display:inline-block; }

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    text-align: center;
    font-family: var(--mono);
}
.metric-val  { font-size: 2rem; color: var(--accent); font-weight: 700; }
.metric-lbl  { font-size: 0.75rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

.header-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
}
.header-title {
    font-family: var(--mono);
    font-size: 1.6rem;
    color: var(--accent);
    letter-spacing: 2px;
}
.header-sub {
    font-size: 0.8rem;
    color: var(--muted);
    font-family: var(--mono);
    letter-spacing: 1px;
}

.log-box {
    background: #050810;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--muted);
    max-height: 160px;
    overflow-y: auto;
}

div[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return sess

def get_model_path():
    # Common locations
    candidates = [
        'inference_model.onnx',
        'bird_detector.onnx',
        '/kaggle/working/outputs/inference_model.onnx',
        '/kaggle/working/output/inference_model.onnx',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess(image: np.ndarray, resolution: int = 560) -> np.ndarray:
    """Resize + letterbox to square, normalise to [0,1], return NCHW float32."""
    h, w = image.shape[:2]
    scale = resolution / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))

    canvas = np.full((resolution, resolution, 3), 114, dtype=np.uint8)
    pad_y  = (resolution - nh) // 2
    pad_x  = (resolution - nw) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]   # HWC → NCHW
    return blob, scale, pad_x, pad_y


def postprocess(outputs, scale, pad_x, pad_y,
                orig_w, orig_h, resolution, conf_thresh):
    """Convert raw ONNX outputs → list of (x1,y1,x2,y2,score) in original coords."""
    logits = outputs[0]   # [1, num_queries, num_classes]
    boxes  = outputs[1]   # [1, num_queries, 4]  cx,cy,w,h normalised

    scores = 1 / (1 + np.exp(-logits[0]))   # sigmoid
    scores = scores.max(axis=-1)             # [num_queries]

    mask   = scores > conf_thresh
    scores = scores[mask]
    boxes  = boxes[0][mask]

    detections = []
    for score, box in zip(scores, boxes):
        cx, cy, bw, bh = box
        # to pixel in padded 560×560 space
        cx *= resolution; cy *= resolution
        bw *= resolution; bh *= resolution
        # remove padding
        cx -= pad_x; cy -= pad_y
        # remove scale
        cx /= scale; cy /= scale
        bw /= scale; bh /= scale

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(orig_w, int(cx + bw / 2))
        y2 = min(orig_h, int(cy + bh / 2))
        detections.append((x1, y1, x2, y2, float(score)))

    return detections


def run_inference(sess, frame_rgb: np.ndarray, resolution: int, conf_thresh: float):
    h, w = frame_rgb.shape[:2]
    blob, scale, pad_x, pad_y = preprocess(frame_rgb, resolution)
    input_name = sess.get_inputs()[0].name
    outputs    = sess.run(None, {input_name: blob})
    return postprocess(outputs, scale, pad_x, pad_y, w, h, resolution, conf_thresh)


def risk_level(n: int):
    if n == 0:
        return 'SAFE',    '✅', 'safe'
    elif n <= 5:
        return 'LOW RISK', '⚠️', 'low'
    else:
        return 'HIGH RISK', '🚨', 'high'


def draw_detections(frame_rgb: np.ndarray, detections):
    img   = frame_rgb.copy()
    color = (0, 229, 255)   # cyan BGR-ish in RGB

    for (x1, y1, x2, y2, score) in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{score:.2f}'
        lx, ly = x1, max(y1 - 6, 0)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (lx, ly - th - 4), (lx + tw + 4, ly + 2), color, -1)
        cv2.putText(img, label, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 14, 23), 1, cv2.LINE_AA)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='font-family:var(--mono);color:var(--accent);font-size:1.1rem;
                letter-spacing:2px;margin-bottom:4px;'>AEROWATCH</div>
    <div style='font-family:var(--mono);color:var(--muted);font-size:0.7rem;
                letter-spacing:1px;margin-bottom:24px;'>BIRD DETECTION SYSTEM v1.0</div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Model")
    model_path_input = st.text_input(
        "ONNX model path",
        value=get_model_path() or 'inference_model.onnx',
        help="Path to your inference_model.onnx file"
    )

    st.markdown("#### 🎛️ Detection")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)
    resolution  = st.select_slider("Input resolution", [416, 480, 560, 640], value=560)

    st.markdown("#### 📹 Video")
    frame_skip = st.slider("Process every Nth frame", 1, 10, 1,
                           help="1 = every frame, 5 = every 5th frame (faster)")

    st.markdown("---")
    st.markdown("""
    <div class='log-box'>
    <div style='color:var(--accent);margin-bottom:6px;'>RISK SCALE</div>
    <div>🟢 0 birds → SAFE</div>
    <div>🟡 1–5 birds → LOW RISK</div>
    <div>🔴 5+ birds → HIGH RISK</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='header-bar'>
  <div>
    <div class='header-title'>🦅 AEROWATCH</div>
    <div class='header-sub'>AIRPORT BIRD DETECTION & RISK ASSESSMENT</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Load model
sess = None
if os.path.exists(model_path_input):
    with st.spinner('Loading ONNX model...'):
        sess = load_model(model_path_input)
    st.success(f'Model loaded from `{model_path_input}`', icon='✅')
else:
    st.warning(
        f'Model not found at `{model_path_input}`. '
        'Update the path in the sidebar.', icon='⚠️'
    )

st.markdown("---")

# ── Mode tabs ─────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["🖼️  Image Detection", "🎬  Video Detection"])


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    uploaded_img = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=['jpg', 'jpeg', 'png'],
        key='img_uploader'
    )

    if uploaded_img and sess:
        pil_img    = Image.open(uploaded_img).convert('RGB')
        frame_rgb  = np.array(pil_img)

        with st.spinner('Running inference...'):
            t0          = time.perf_counter()
            detections  = run_inference(sess, frame_rgb, resolution, conf_thresh)
            elapsed_ms  = (time.perf_counter() - t0) * 1000

        annotated   = draw_detections(frame_rgb, detections)
        n_birds     = len(detections)
        label, icon, cls = risk_level(n_birds)

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-val'>{n_birds}</div>
              <div class='metric-lbl'>Birds Detected</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-val'>{elapsed_ms:.0f}<span style='font-size:1rem'>ms</span></div>
              <div class='metric-lbl'>Inference Time</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            avg_score = np.mean([d[4] for d in detections]) if detections else 0
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-val'>{avg_score:.2f}</div>
              <div class='metric-lbl'>Avg Confidence</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-val'>{icon}</div>
              <div class='metric-lbl'>Risk Level</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk badge
        st.markdown(f"<div class='risk-{cls}'>{icon} &nbsp; {label}</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-side original vs annotated
        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(pil_img, use_container_width=True)
        with col_ann:
            st.markdown("**Detections**")
            st.image(annotated, use_container_width=True)

        # Download annotated image
        annotated_pil = Image.fromarray(annotated)
        import io
        buf = io.BytesIO()
        annotated_pil.save(buf, format='JPEG', quality=92)
        st.download_button(
            '⬇️  Download annotated image',
            data=buf.getvalue(),
            file_name='bird_detection_result.jpg',
            mime='image/jpeg'
        )

    elif uploaded_img and not sess:
        st.error('Model not loaded — fix the model path in the sidebar.')
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:var(--muted);
                    font-family:var(--mono);font-size:0.9rem;'>
            Upload an image to begin detection
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    uploaded_vid = st.file_uploader(
        "Upload a video (MP4 / AVI / MOV)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key='vid_uploader'
    )

    if uploaded_vid and sess:
        # Save to temp file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25
        duration_s   = total_frames / fps_src

        st.markdown(f"""
        <div class='log-box' style='margin-bottom:16px;'>
          <span style='color:var(--accent);'>VIDEO INFO</span> &nbsp;|&nbsp;
          {total_frames} frames &nbsp;|&nbsp;
          {fps_src:.1f} fps &nbsp;|&nbsp;
          {duration_s:.1f}s duration
        </div>""", unsafe_allow_html=True)

        run_btn = st.button('▶  Start Detection', use_container_width=True)

        if run_btn:
            # UI placeholders
            frame_display  = st.empty()
            metrics_ph     = st.empty()
            progress_bar   = st.progress(0)
            status_ph      = st.empty()

            frame_idx    = 0
            max_birds    = 0
            total_det    = 0
            processed    = 0
            inference_times = []

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

                # Skip frames
                if frame_idx % frame_skip != 0:
                    continue

                frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                t0         = time.perf_counter()
                detections = run_inference(sess, frame_rgb, resolution, conf_thresh)
                inf_ms     = (time.perf_counter() - t0) * 1000
                inference_times.append(inf_ms)

                n_birds  = len(detections)
                max_birds = max(max_birds, n_birds)
                total_det += n_birds
                processed += 1

                annotated = draw_detections(frame_rgb, detections)

                # Overlay: frame counter + risk
                risk_lbl, risk_icon, risk_cls = risk_level(n_birds)
                risk_colors = {'safe': (0,230,118), 'low': (255,179,0), 'high': (255,23,68)}
                rc = risk_colors[risk_cls]
                cv2.putText(annotated,
                            f'Frame {frame_idx}/{total_frames}  |  {n_birds} birds  |  {risk_lbl}',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rc, 2, cv2.LINE_AA)
                cv2.putText(annotated,
                            f'{inf_ms:.0f}ms  |  conf>{conf_thresh}',
                            (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1, cv2.LINE_AA)

                frame_display.image(annotated, use_container_width=True)

                avg_inf = np.mean(inference_times)
                metrics_ph.markdown(f"""
                <div style='display:flex;gap:12px;margin-top:8px;'>
                  <div class='metric-box' style='flex:1'>
                    <div class='metric-val'>{n_birds}</div>
                    <div class='metric-lbl'>Birds (this frame)</div>
                  </div>
                  <div class='metric-box' style='flex:1'>
                    <div class='metric-val'>{max_birds}</div>
                    <div class='metric-lbl'>Peak Count</div>
                  </div>
                  <div class='metric-box' style='flex:1'>
                    <div class='metric-val'>{avg_inf:.0f}<span style='font-size:1rem'>ms</span></div>
                    <div class='metric-lbl'>Avg Inference</div>
                  </div>
                  <div class='metric-box' style='flex:1'>
                    <div class='metric-val'>{risk_icon}</div>
                    <div class='metric-lbl'>{risk_lbl}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                status_ph.markdown(
                    f"Processing frame **{frame_idx}** / {total_frames} "
                    f"— {processed} frames analysed"
                )

            cap.release()
            os.unlink(tfile.name)
            progress_bar.progress(1.0)

            # Final summary
            overall_risk, overall_icon, overall_cls = risk_level(max_birds)
            st.markdown("---")
            st.markdown("### 📊 Session Summary")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown(f"""
                <div class='metric-box'>
                  <div class='metric-val'>{processed}</div>
                  <div class='metric-lbl'>Frames Analysed</div>
                </div>""", unsafe_allow_html=True)
            with cc2:
                st.markdown(f"""
                <div class='metric-box'>
                  <div class='metric-val'>{max_birds}</div>
                  <div class='metric-lbl'>Peak Bird Count</div>
                </div>""", unsafe_allow_html=True)
            with cc3:
                st.markdown(f"""
                <div class='metric-box'>
                  <div class='metric-val'>{np.mean(inference_times):.0f}ms</div>
                  <div class='metric-lbl'>Avg Inference</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='risk-{overall_cls}'>{overall_icon} &nbsp;"
                f"Overall: {overall_risk} — peak {max_birds} birds detected</div>",
                unsafe_allow_html=True
            )

    elif uploaded_vid and not sess:
        st.error('Model not loaded — fix the model path in the sidebar.')
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:var(--muted);
                    font-family:var(--mono);font-size:0.9rem;'>
            Upload a video to begin detection
        </div>""", unsafe_allow_html=True)
