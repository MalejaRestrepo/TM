import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# CONFIGURACIÓN GENERAL
st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="🧠",
    layout="centered"
)

# 🎨 ESTILO LAVANDA-VIOLETA
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #e8dcff 0%, #d7c4ff 100%);
        color: #22143d;
        font-family: 'Poppins', sans-serif;
    }

    .block-container {
        background: #faf7ff;
        border: 1px solid #cbb3ff;
        border-radius: 16px;
        padding: 2rem 2.2rem;
        box-shadow: 0 10px 24px rgba(34, 20, 61, 0.12);
    }

    h1, h2, h3, h4 {
        color: #3b2168;
        text-align: center;
        font-weight: 700;
    }

    p, li, label {
        color: #22143d;
        font-size: 15px;
    }

    section[data-testid="stSidebar"] {
        background: #efe6ff;
        border-right: 2px solid #c9b1ff;
        color: #2a1d5c;
    }

    section[data-testid="stSidebar"] * {
        color: #2a1d5c !important;
        font-size: 15px;
    }

    div.stButton > button {
        background-color: #8b6aff;
        color: white !important;
        font-weight: 700;
        border-radius: 10px;
        border: 1px solid #6f51ea;
        box-shadow: 0 6px 14px rgba(34, 20, 61, 0.18);
        font-size: 16px;
        padding: 9px 24px;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background-color: #6f51ea;
        transform: translateY(-1px);
    }

    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #5a3ccf 0%, #7b59e3 100%) !important;
        color: white !important;
        height: 3.5rem;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.25);
    }

    audio, img {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    </style>
""", unsafe_allow_html=True)

# Muestra la versión de Python
st.caption(f"🐍 Versión de Python: {platform.python_version()}")

# CARGA DEL MODELO
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# TÍTULO E IMAGEN PRINCIPAL
st.title("🔍 Reconocimiento de Imágenes con Teachable Machine")
image = Image.open('cinna3.jpeg')
st.image(image, width=350)

# SIDEBAR
with st.sidebar:
    st.subheader("🧠 Instrucciones")
    st.write("Usa un modelo entrenado en **Teachable Machine** para identificar objetos en tus fotos.")
    st.write("Presiona el botón de cámara para capturar una imagen y ver el resultado.")

# CAPTURA DE IMAGEN
img_file_buffer = st.camera_input("📸 Toma una Foto")

if img_file_buffer is not None:
    # Procesamiento de imagen
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)

    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Inferencia
    prediction = model.predict(data)
    print(prediction)

    # RESULTADOS
    st.markdown("---")
    st.subheader("🎯 Resultado de la Predicción:")

    if prediction[0][0] > 0.5:
        st.success(f"➡️ Izquierda — **Probabilidad:** {prediction[0][0]:.3f}")
    elif prediction[0][1] > 0.5:
        st.success(f"⬆️ Arriba — **Probabilidad:** {prediction[0][1]:.3f}")
    else:
        st.info("❓ No se detectó una categoría con suficiente confianza.")

# PIE DE PÁGINA
st.markdown("---")
st.caption("💜 Desarrollado con Streamlit + Teachable Machine")
