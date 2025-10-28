import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# 🌸 Estilos personalizados (lavanda + violeta)
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background-color: #EADCF8;
        color: #2E1A47;
        font-family: 'Poppins', sans-serif;
    }

    /* Título principal */
    h1 {
        color: #5B3EA1;
        text-align: center;
        font-weight: 700;
    }

    /* Subtítulos */
    h2, h3 {
        color: #6A42C2;
    }

    /* Botones */
    div.stButton > button {
        background-color: #7B5CD6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #5B3EA1;
        color: #fff;
        transform: scale(1.03);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #D8C3F1;
        color: #2E1A47;
    }
    </style>
""", unsafe_allow_html=True)

# Información del entorno
st.write("Versión de Python:", platform.python_version())

# Cargar modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título e imagen principal
st.title("💜 Reconocimiento de Imágenes")

image = Image.open('cinna3.jpeg')
st.image(image, width=350)

with st.sidebar:
    st.subheader("🧠 Identificador de imágenes")
    st.write("Usa un modelo entrenado en Teachable Machine para reconocer objetos o gestos.")
    st.info("Toma una foto o carga una imagen para probar el modelo.")

# Captura de imagen
img_file_buffer = st.camera_input("📷 Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    # Ajuste de tamaño
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalización
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)

    # Resultados
    st.markdown("---")
    st.subheader("📊 Resultados del modelo")

    if prediction[0][0] > 0.5:
        st.success(f"Izquierda → Probabilidad: **{prediction[0][0]:.2f}**")
    if prediction[0][1] > 0.5:
        st.info(f"Arriba → Probabilidad: **{prediction[0][1]:.2f}**")
    # Si el modelo tiene más salidas, pueden agregarse aquí sin alterar la lógica base
