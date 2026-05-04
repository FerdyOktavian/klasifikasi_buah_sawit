import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# =========================
# KONFIGURASI
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_mobilenetv2_sawit_finetuned.keras"
CLASS_NAMES = ["belum_masak", "masak", "terlalu_masak"]
DISPLAY_NAMES = {
    "belum_masak": "Belum Masak",
    "masak": "Masak",
    "terlalu_masak": "Terlalu Masak"
}
IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# FUNGSI PREDIKSI
# =========================
def predict_image(input_image):
    image = Image.open(input_image).convert("RGB")
    image_resized = image.resize(IMG_SIZE)
    img_array = np.array(image_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch, verbose=0)[0]
    pred_index = np.argmax(predictions)
    pred_class = CLASS_NAMES[pred_index]
    confidence = predictions[pred_index] * 100
    return image, pred_class, confidence, predictions

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Klasifikasi Kematangan Sawit",
    page_icon="🌴",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0F1117;
    --surface:   #171B26;
    --surface-2: #1E2436;
    --border:    rgba(255,255,255,0.09);
    --border-2:  rgba(255,255,255,0.15);
    --gold:      #C9A84C;
    --gold-dim:  rgba(201,168,76,0.15);
    --gold-pale: rgba(201,168,76,0.08);
    --text:      #F0EDE6;
    --text-2:    #A8A39A;
    --text-3:    #6B6760;
    --green:     #4A7C5F;
    --green-pale: rgba(74,124,95,0.15);
    --red-pale:  rgba(180,70,60,0.15);
    --red:       #C96B5A;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── NAVBAR ── */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    height: 68px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 60px;
    background: rgba(15,17,23,0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
}

.nav-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.2px;
}

.nav-brand em {
    font-style: normal;
    color: var(--gold);
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 36px;
}

.nav-links a {
    font-size: 15px;
    font-weight: 400;
    color: var(--text-2) !important;
    text-decoration: none;
    letter-spacing: 0.3px;
    transition: color 0.2s;
}

.nav-links a:hover { color: var(--text) !important; }

.nav-badge {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--gold);
    border: 1px solid var(--gold);
    border-radius: 4px;
    padding: 6px 14px;
    opacity: 0.85;
}

.spacer { height: 68px; }

/* ── HERO ── */
.hero {
    padding: 110px 60px 100px;
    border-bottom: 1px solid var(--border);
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 80px;
    align-items: center;
    background:
        radial-gradient(ellipse 60% 50% at 80% 50%, rgba(201,168,76,0.06) 0%, transparent 70%),
        var(--bg);
}

.hero-eyebrow {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 24px;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 64px;
    font-weight: 700;
    line-height: 1.08;
    color: var(--text);
    letter-spacing: -0.5px;
    margin-bottom: 28px;
}

.hero-title em {
    font-style: italic;
    color: var(--gold);
}

.hero-body {
    font-size: 18px;
    font-weight: 300;
    line-height: 1.85;
    color: var(--text-2);
    max-width: 460px;
    margin-bottom: 52px;
}

.hero-rule {
    width: 100%;
    height: 1px;
    background: var(--border);
    margin-bottom: 36px;
}

.hero-stats {
    display: flex;
    gap: 48px;
}

.stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 42px;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
    margin-bottom: 6px;
}

.stat-label {
    font-size: 13px;
    font-weight: 400;
    color: var(--text-3);
    letter-spacing: 0.3px;
}

/* Class cards on hero right */
.class-stack {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.class-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px 28px;
    display: flex;
    align-items: flex-start;
    gap: 18px;
    transition: border-color 0.25s;
}

.class-card:hover { border-color: var(--border-2); }

.class-marker {
    width: 4px;
    border-radius: 4px;
    flex-shrink: 0;
    align-self: stretch;
    min-height: 48px;
}

.marker-red    { background: #C96B5A; }
.marker-green  { background: var(--green); }
.marker-gold   { background: var(--gold); }

.class-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
}

.class-desc {
    font-size: 14px;
    font-weight: 300;
    line-height: 1.7;
    color: var(--text-2);
}

/* ── SECTION WRAPPER ── */
.sec {
    padding: 100px 60px;
    border-bottom: 1px solid var(--border);
}

.sec-alt { background: var(--surface); }

.sec-eyebrow {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 16px;
}

.sec-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 48px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 18px;
}

.sec-desc {
    font-size: 17px;
    font-weight: 300;
    line-height: 1.85;
    color: var(--text-2);
    max-width: 580px;
    margin-bottom: 56px;
}

/* ── FILE UPLOADER ── */
div[data-testid="stFileUploader"] {
    background: var(--surface-2) !important;
    border: 1.5px dashed rgba(201,168,76,0.5) !important;
    border-radius: 14px !important;
    padding: 28px !important;
}

div[data-testid="stFileUploader"] label {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #black !important;
}

div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] p {
    font-size: 14px !important;
    color: #black !important;
}

div[data-testid="stFileUploader"] button {
    background: var(--gold) !important;
    color: #0F1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
}

/* ── RADIO ── */
.stRadio > label {
    font-size: 17px !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}

.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 16px !important;
    color: var(--text) !important;
}

/* ── RESULT PANEL ── */
.result-panel {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 44px;
}

.result-eyebrow {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 14px;
}

.result-class {
    font-family: 'Cormorant Garamond', serif;
    font-size: 52px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 12px;
}

.result-conf {
    font-size: 18px;
    font-weight: 300;
    color: var(--text-2);
    margin-bottom: 40px;
}

.result-conf strong {
    color: var(--gold);
    font-size: 24px;
    font-weight: 700;
    font-family: 'Cormorant Garamond', serif;
}

.result-divider {
    height: 1px;
    background: var(--border);
    margin: 0 0 32px 0;
}

.prob-heading {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 24px;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.prob-name {
    font-size: 16px;
    font-weight: 400;
    color: var(--text-2);
}

.prob-pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--gold);
}

/* Progress bar */
.stProgress > div > div > div {
    background: var(--surface) !important;
    height: 8px !important;
    border-radius: 999px !important;
}

.stProgress > div > div > div > div {
    background: var(--gold) !important;
    border-radius: 999px !important;
}

.result-note {
    font-size: 14px;
    font-weight: 300;
    color: var(--text-3);
    line-height: 1.75;
    margin-top: 28px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
}

.empty-state {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 72px 40px;
    text-align: center;
}

.empty-icon { font-size: 52px; margin-bottom: 20px; }

.empty-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 12px;
}

.empty-text {
    font-size: 16px;
    font-weight: 300;
    color: var(--text-2);
    line-height: 1.7;
}

/* ── CAMERA INPUT ── */
div[data-testid="stCameraInput"] label {
    font-size: 17px !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}

/* ── STEPS GRID ── */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 2px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}

.step-cell {
    background: var(--surface);
    padding: 44px 32px;
}

.step-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 56px;
    font-weight: 700;
    color: rgba(201,168,76,0.18);
    line-height: 1;
    margin-bottom: 22px;
}

.step-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 12px;
}

.step-text {
    font-size: 15px;
    font-weight: 300;
    line-height: 1.8;
    color: var(--text-2);
}

/* ── SISTEM ── */
.sistem-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

.sistem-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 44px 36px;
    transition: border-color 0.25s;
}

.sistem-card:hover { border-color: var(--border-2); }

.sistem-icon {
    font-size: 32px;
    margin-bottom: 22px;
    display: block;
}

.sistem-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 14px;
}

.sistem-text {
    font-size: 15px;
    font-weight: 300;
    line-height: 1.85;
    color: var(--text-2);
}

/* ── BIO ── */
.bio-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 80px;
    align-items: start;
}

.bio-text {
    font-size: 17px;
    font-weight: 300;
    line-height: 1.9;
    color: var(--text-2);
    margin-bottom: 18px;
}

.bio-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 32px;
}

.bio-tag {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--gold);
    border: 1px solid rgba(201,168,76,0.4);
    border-radius: 4px;
    padding: 7px 16px;
    background: var(--gold-pale);
}

.bio-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}

.bio-item {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 22px;
}

.bio-key {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 9px;
}

.bio-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
}

/* ── KONTAK ── */
.kontak-layout {
    display: grid;
    grid-template-columns: 0.55fr 1fr;
    gap: 80px;
    align-items: start;
}

.kontak-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.kontak-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 26px;
}

.kontak-key {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-3);
}

.kontak-val {
    font-size: 16px;
    font-weight: 400;
    color: var(--text);
}

/* ── FOOTER ── */
.footer {
    padding: 40px 60px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--surface);
    border-top: 1px solid var(--border);
}

.footer-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
}

.footer-brand em {
    font-style: normal;
    color: var(--gold);
}

.footer-copy {
    font-size: 14px;
    font-weight: 300;
    color: var(--text-3);
}

/* ── IMAGE ── */
img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-top-color: var(--gold) !important;
}

/* Alerts */
.stAlert {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}

/* ── RESPONSIVE ── */
@media (max-width: 960px) {
    .navbar { padding: 0 24px; }
    .nav-links { display: none; }
    .hero { grid-template-columns: 1fr; padding: 80px 24px 60px; }
    .hero-title { font-size: 44px; }
    .sec { padding: 72px 24px; }
    .steps-grid { grid-template-columns: 1fr 1fr; }
    .sistem-grid { grid-template-columns: 1fr; }
    .bio-layout { grid-template-columns: 1fr; gap: 40px; }
    .kontak-layout { grid-template-columns: 1fr; gap: 32px; }
    .footer { flex-direction: column; gap: 12px; text-align: center; padding: 32px 24px; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════
st.markdown("""
<div class="navbar">
    <div class="nav-brand">Sawit<em>Vision</em></div>
    <div class="nav-links">
        <a href="#prediksi">Prediksi</a>
        <a href="#cara-pakai">Cara Pakai</a>
        <a href="#sistem">Sistem</a>
        <a href="#biografi">Biografi</a>
        <a href="#kontak">Kontak</a>
    </div>
    <div class="nav-badge">AI · CNN</div>
</div>
<div class="spacer"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# HERO
# ══════════════════════════════════════
st.markdown("""
<div class="hero">
    <div>
        <div class="hero-eyebrow">Sistem Klasifikasi Berbasis Deep Learning</div>
        <div class="hero-title">
            Identifikasi Kematangan<br>
            Buah <em>Kelapa Sawit</em>
        </div>
        <div class="hero-body">
            Sistem berbasis MobileNetV2 yang mampu mengidentifikasi tingkat
            kematangan buah kelapa sawit secara akurat hanya dari sebuah foto.
            Cepat, tepat, dan mudah digunakan.
        </div>
        <div class="hero-rule"></div>
        <div class="hero-stats">
            <div>
                <div class="stat-num">3</div>
                <div class="stat-label">Kelas Kematangan</div>
            </div>
            <div>
                <div class="stat-num">CNN</div>
                <div class="stat-label">Metode</div>
            </div>
            <div>
                <div class="stat-num">AI</div>
                <div class="stat-label">Deep Learning</div>
            </div>
        </div>
    </div>
    <div class="class-stack">
        <div class="class-card">
            <div class="class-marker marker-red"></div>
            <div>
                <div class="class-title">Belum Masak</div>
                <div class="class-desc">Warna cenderung gelap atau kehijauan. Kandungan minyak masih rendah dan belum siap panen.</div>
            </div>
        </div>
        <div class="class-card">
            <div class="class-marker marker-green"></div>
            <div>
                <div class="class-title">Masak</div>
                <div class="class-desc">Warna matang optimal, kandungan minyak tertinggi. Kondisi ideal untuk dipanen.</div>
            </div>
        </div>
        <div class="class-card">
            <div class="class-marker marker-gold"></div>
            <div>
                <div class="class-title">Terlalu Masak</div>
                <div class="class-desc">Kematangan berlebih dengan tanda brondolan lepas. Kualitas minyak mulai menurun.</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# PREDIKSI
# ══════════════════════════════════════
st.markdown('<div id="prediksi"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec sec-alt">
    <div class="sec-eyebrow">Upload &amp; Analisis</div>
    <div class="sec-title">Mulai Prediksi</div>
    <div class="sec-desc">
        Upload gambar atau ambil foto langsung. Sistem akan menganalisis dan
        menampilkan hasil klasifikasi secara otomatis.
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div style="background: var(--surface, #171B26); padding: 0 60px 80px;">', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        input_mode = st.radio(
            "Metode Input",
            ["Upload Gambar", "Ambil Foto Langsung"],
            horizontal=True
        )

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        uploaded_file = None
        camera_file = None

        if input_mode == "Upload Gambar":
            uploaded_file = st.file_uploader(
                "Pilih Gambar Buah Sawit",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
        else:
            camera_file = st.camera_input("Ambil foto buah kelapa sawit")

        input_file = uploaded_file if uploaded_file is not None else camera_file

        if input_file is not None:
            image, pred_class, confidence, predictions = predict_image(input_file)
            st.markdown('<div style="margin-top: 24px;">', unsafe_allow_html=True)
            st.image(image, caption="Gambar yang dianalisis", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        if input_file is not None:
            st.markdown(f"""
            <div class="result-panel">
                <div class="result-eyebrow">Hasil Klasifikasi</div>
                <div class="result-class">{DISPLAY_NAMES[pred_class]}</div>
                <div class="result-conf">
                    Tingkat keyakinan model: <strong>{confidence:.1f}%</strong>
                </div>
                <div class="result-divider"></div>
                <div class="prob-heading">Distribusi Probabilitas</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="result-panel" style="margin-top: 2px; border-top: none; border-radius: 0 0 16px 16px; padding-top: 0;">', unsafe_allow_html=True)

            for class_name, prob in zip(CLASS_NAMES, predictions):
                st.markdown(f"""
                <div class="prob-row">
                    <span class="prob-name">{DISPLAY_NAMES[class_name]}</span>
                    <span class="prob-pct">{prob * 100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob))
                st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

            if confidence < 70:
                st.markdown("""
                <div style="background: rgba(201,107,90,0.12); border: 1px solid rgba(201,107,90,0.35);
                            border-radius: 10px; padding: 16px 20px; margin-top: 16px;
                            font-size: 15px; color: #E8A898; font-weight: 300; line-height: 1.7;">
                    ⚠️ Confidence masih rendah. Coba gunakan gambar dengan pencahayaan lebih baik
                    dan pastikan objek buah terlihat jelas.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="result-note">
                Hasil prediksi dapat dipengaruhi oleh pencahayaan, kualitas gambar,
                sudut pengambilan, dan kemiripan visual antar kelas.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🌿</div>
                <div class="empty-title">Belum Ada Gambar</div>
                <div class="empty-text">
                    Upload gambar atau ambil foto buah kelapa sawit<br>
                    untuk memulai analisis klasifikasi.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════
# CARA PAKAI
# ══════════════════════════════════════
st.markdown('<div id="cara-pakai"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec">
    <div class="sec-eyebrow">Panduan</div>
    <div class="sec-title">Cara Menggunakan</div>
    <div class="sec-desc">
        Empat langkah mudah untuk mendapatkan hasil klasifikasi kematangan buah sawit.
    </div>
    <div class="steps-grid">
        <div class="step-cell">
            <div class="step-num">01</div>
            <div class="step-title">Siapkan Gambar</div>
            <div class="step-text">Gunakan foto buah sawit dengan pencahayaan merata dan objek terlihat jelas. Pastikan gambar tidak buram atau terlalu gelap.</div>
        </div>
        <div class="step-cell">
            <div class="step-num">02</div>
            <div class="step-title">Pilih Metode</div>
            <div class="step-text">Pilih antara upload gambar dari galeri atau ambil foto langsung menggunakan kamera perangkat Anda.</div>
        </div>
        <div class="step-cell">
            <div class="step-num">03</div>
            <div class="step-title">Analisis Otomatis</div>
            <div class="step-text">Sistem secara otomatis memproses gambar menggunakan model deep learning MobileNetV2 tanpa perlu klik tambahan.</div>
        </div>
        <div class="step-cell">
            <div class="step-num">04</div>
            <div class="step-title">Baca Hasilnya</div>
            <div class="step-text">Dapatkan kelas prediksi, nilai confidence, dan distribusi probabilitas lengkap untuk setiap kategori kematangan.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# SISTEM
# ══════════════════════════════════════
st.markdown('<div id="sistem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec sec-alt">
    <div class="sec-eyebrow">Teknologi</div>
    <div class="sec-title">Tentang Sistem</div>
    <div class="sec-desc">
        Dibangun menggunakan pendekatan deep learning mutakhir untuk klasifikasi citra secara akurat dan efisien.
    </div>
    <div class="sistem-grid">
        <div class="sistem-card">
            <span class="sistem-icon">🧠</span>
            <div class="sistem-title">Arsitektur Model</div>
            <div class="sistem-text">
                Menggunakan MobileNetV2, arsitektur CNN yang ringan namun akurat. Dioptimalkan
                dengan teknik fine-tuning untuk klasifikasi kematangan buah sawit.
            </div>
        </div>
        <div class="sistem-card">
            <span class="sistem-icon">📷</span>
            <div class="sistem-title">Input Sistem</div>
            <div class="sistem-text">
                Menerima citra digital buah kelapa sawit melalui upload file maupun
                kamera langsung. Format yang didukung: JPG, JPEG, dan PNG.
            </div>
        </div>
        <div class="sistem-card">
            <span class="sistem-icon">📊</span>
            <div class="sistem-title">Output Sistem</div>
            <div class="sistem-text">
                Menghasilkan prediksi kelas kematangan, nilai confidence model, dan distribusi
                probabilitas lengkap untuk ketiga kategori secara real-time.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# BIOGRAFI
# ══════════════════════════════════════
st.markdown('<div id="biografi"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec">
    <div class="sec-eyebrow">Pengembang</div>
    <div class="sec-title">Biografi Pembuat</div>
    <div class="bio-layout">
        <div>
            <div class="bio-text">
                Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi di bidang
                kecerdasan buatan dan pengolahan citra digital. Tujuan utamanya adalah
                membantu proses identifikasi tingkat kematangan buah kelapa sawit secara
                lebih objektif menggunakan teknologi deep learning.
            </div>
            <div class="bio-text">
                Sistem dirancang agar mudah diakses melalui web oleh siapa saja, tanpa
                memerlukan keahlian teknis khusus di bidang kecerdasan buatan.
            </div>
            <div class="bio-tags">
                <span class="bio-tag">Artificial Intelligence</span>
                <span class="bio-tag">Computer Vision</span>
                <span class="bio-tag">Deep Learning</span>
                <span class="bio-tag">Skripsi</span>
            </div>
        </div>
        <div class="bio-grid">
            <div class="bio-item">
                <div class="bio-key">Nama</div>
                <div class="bio-val">Muhammad Ferdy</div>
            </div>
            <div class="bio-item">
                <div class="bio-key">Bidang</div>
                <div class="bio-val">Computer Vision</div>
            </div>
            <div class="bio-item">
                <div class="bio-key">Topik</div>
                <div class="bio-val">Klasifikasi Sawit</div>
            </div>
            <div class="bio-item">
                <div class="bio-key">Tahun</div>
                <div class="bio-val">2026</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# KONTAK
# ══════════════════════════════════════
st.markdown('<div id="kontak"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec sec-alt">
    <div class="kontak-layout">
        <div>
            <div class="sec-eyebrow">Hubungi Kami</div>
            <div class="sec-title">Kontak<br>Pembuat</div>
        </div>
        <div class="kontak-list">
            <div class="kontak-row">
                <span class="kontak-key">Nama</span>
                <span class="kontak-val">Muhammad Ferdy</span>
            </div>
            <div class="kontak-row">
                <span class="kontak-key">Email</span>
                <span class="kontak-val">isi_email_kamu@gmail.com</span>
            </div>
            <div class="kontak-row">
                <span class="kontak-key">Instagram</span>
                <span class="kontak-val">@username_kamu</span>
            </div>
            <div class="kontak-row">
                <span class="kontak-key">Universitas</span>
                <span class="kontak-val">isi_nama_kampus_kamu</span>
            </div>
            <div class="kontak-row">
                <span class="kontak-key">Program Studi</span>
                <span class="kontak-val">isi_program_studi_kamu</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# FOOTER
# ══════════════════════════════════════
st.markdown("""
<div class="footer">
    <div class="footer-brand">Sawit<em>Vision</em> AI</div>
    <div class="footer-copy">© 2026 — Sistem Klasifikasi Kematangan Buah Kelapa Sawit Berbasis MobileNetV2</div>
</div>
""", unsafe_allow_html=True)
