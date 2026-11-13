"""
Clasificador de Imágenes COVID-19
Aplicación académica en Streamlit que usa un modelo de Teachable Machine
(keras_model.h5 + labels.txt) para clasificar tomografías computarizadas.
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador de Imágenes COVID-19",
    page_icon="🫁",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
IMG_SIZE = (224, 224)  # Tamaño de entrada de los modelos de Teachable Machine

# El archivo labels.txt exportado contiene nombres genéricos ("Class 1", "Class 2").
# Esta lista asigna el nombre real a cada clase según su posición en labels.txt:
#   posición 0 -> "Class 1"  |  posición 1 -> "Class 2"
# Si en Teachable Machine creó las clases en el orden contrario, intercambie los dos
# nombres de esta lista.
NOMBRES_CLASES = ["COVID 19", "NO COVID 19"]


# ---------------------------------------------------------------------------
# Carga del modelo y de las clases (se hace una sola vez)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el modelo, por favor espere...")
def cargar_modelo():
    # Importación diferida para que la interfaz cargue rápido
    import tensorflow as tf

    # compile=False: solo se necesita para inferencia
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


@st.cache_data
def cargar_clases():
    """Lee labels.txt. Formato típico de Teachable Machine: '0 COVID 19'."""
    clases = []
    with open(LABELS_PATH, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            partes = linea.split(" ", 1)
            if len(partes) == 2 and partes[0].isdigit():
                clases.append(partes[1].strip())
            else:
                clases.append(linea)

    # Si labels.txt tiene nombres genéricos ("Class 1", "Class 2"), se usan
    # los nombres definidos en NOMBRES_CLASES.
    genericas = all(c.lower().startswith("class") for c in clases)
    if genericas and len(clases) == len(NOMBRES_CLASES):
        return list(NOMBRES_CLASES)
    return clases


# ---------------------------------------------------------------------------
# Preprocesamiento y predicción
# ---------------------------------------------------------------------------
def preprocesar(imagen: Image.Image) -> np.ndarray:
    """Preprocesamiento idéntico al de Teachable Machine:
    recorte/redimensión a 224x224 y normalización al rango [-1, 1]."""
    imagen = imagen.convert("RGB")
    imagen = ImageOps.fit(imagen, IMG_SIZE, Image.Resampling.LANCZOS)
    arreglo = np.asarray(imagen, dtype=np.float32)
    arreglo = (arreglo / 127.5) - 1.0
    return np.expand_dims(arreglo, axis=0)  # forma (1, 224, 224, 3)


def predecir(modelo, imagen: Image.Image) -> np.ndarray:
    entrada = preprocesar(imagen)
    salida = modelo.predict(entrada, verbose=0)
    return salida[0]


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.title("🫁 Clasificador de Imágenes COVID-19")
st.write(
    "Cargue una imagen de tomografía computarizada para clasificarla como "
    "**COVID 19** o **NO COVID 19** mediante un modelo entrenado con "
    "Teachable Machine."
)

st.warning(
    "⚠️ **Aviso importante:** esta aplicación es exclusivamente académica y "
    "**no constituye un diagnóstico médico**. Ante cualquier duda sobre su "
    "salud, consulte a un profesional sanitario."
)

# Verificación de archivos del proyecto
faltantes = [p.name for p in (MODEL_PATH, LABELS_PATH) if not p.exists()]
if faltantes:
    st.error(
        "No se encontraron los siguientes archivos en la carpeta del proyecto: "
        + ", ".join(faltantes)
        + ". Colóquelos junto a app.py e intente nuevamente."
    )
    st.stop()

try:
    modelo = cargar_modelo()
    clases = cargar_clases()
except Exception as error:
    st.error(f"No se pudo cargar el modelo: {error}")
    st.stop()

archivo = st.file_uploader(
    "Seleccione una imagen (JPG, JPEG o PNG)",
    type=["jpg", "jpeg", "png"],
)

if archivo is not None:
    try:
        imagen = Image.open(archivo)
        imagen.load()
    except Exception:
        st.error("No se pudo leer la imagen. Intente con otro archivo.")
        st.stop()

    st.subheader("Imagen seleccionada")
    st.image(imagen, caption=archivo.name)

    with st.spinner("Clasificando la imagen..."):
        probabilidades = predecir(modelo, imagen)

    if len(probabilidades) != len(clases):
        st.error(
            "El número de clases en labels.txt no coincide con la salida del modelo."
        )
        st.stop()

    indice = int(np.argmax(probabilidades))
    clase_predicha = clases[indice]
    confianza = float(probabilidades[indice]) * 100

    st.subheader("Resultado de la clasificación")
    mensaje = f"Clasificación: **{clase_predicha}** (confianza: {confianza:.2f} %)"
    if clase_predicha.upper().startswith("NO"):
        st.success(mensaje)
    else:
        st.error(mensaje)

    st.subheader("Porcentajes de confianza")
    for nombre, prob in zip(clases, probabilidades):
        porcentaje = float(prob) * 100
        st.write(f"**{nombre}:** {porcentaje:.2f} %")
        st.progress(min(max(float(prob), 0.0), 1.0))
else:
    st.info("Esperando una imagen para clasificar.")

st.divider()
st.caption(
    "Proyecto académico · Modelo entrenado con Teachable Machine · "
    "No sustituye la evaluación de un profesional médico."
)
