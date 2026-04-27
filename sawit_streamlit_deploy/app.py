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

    # Tidak perlu preprocess_input lagi
    # karena preprocessing sudah dimasukkan ke dalam model saat training
    predictions = model.predict(img_batch, verbose=0)[0]

    pred_index = np.argmax(predictions)
    pred_class = CLASS_NAMES[pred_index]
    confidence = predictions[pred_index] * 100

    return image, pred_class, confidence, predictions


# =========================
# TAMPILAN WEB
# =========================
st.set_page_config(
    page_title="Klasifikasi Kematangan Sawit",
    page_icon="🌴",
    layout="centered"
)

st.title("🌴 Klasifikasi Kematangan Buah Kelapa Sawit")

st.write(
    "Aplikasi ini digunakan untuk mengklasifikasikan tingkat kematangan "
    "buah kelapa sawit berdasarkan citra digital menggunakan model CNN "
    "dengan arsitektur MobileNetV2."
)

st.divider()

st.subheader("Input Gambar")

input_mode = st.radio(
    "Pilih metode input:",
    ["Upload Gambar", "Ambil Foto Langsung"]
)

uploaded_file = None
camera_file = None

if input_mode == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar buah kelapa sawit",
        type=["jpg", "jpeg", "png"]
    )
else:
    camera_file = st.camera_input("Ambil foto buah kelapa sawit")

input_file = uploaded_file if uploaded_file is not None else camera_file

if input_file is not None:
    image, pred_class, confidence, predictions = predict_image(input_file)

    st.image(image, caption="Gambar yang diproses", use_container_width=True)

    st.subheader("Hasil Prediksi")

    st.success(f"Prediksi: **{DISPLAY_NAMES[pred_class]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    if confidence < 70:
        st.warning(
            "Confidence masih rendah. Coba gunakan gambar dengan pencahayaan "
            "yang lebih baik dan pastikan objek buah terlihat jelas."
        )

    st.subheader("Detail Probabilitas")

    for class_name, prob in zip(CLASS_NAMES, predictions):
        st.write(f"{DISPLAY_NAMES[class_name]}: {prob * 100:.2f}%")
        st.progress(float(prob))

    st.divider()

    st.caption(
        "Catatan: Hasil prediksi dapat dipengaruhi oleh pencahayaan, kualitas gambar, "
        "sudut pengambilan gambar, dan kemiripan visual antar kelas."
    )

else:
    st.warning("Silakan upload gambar atau ambil foto terlebih dahulu.")
