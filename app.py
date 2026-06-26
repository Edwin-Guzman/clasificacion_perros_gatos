import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración de la interfaz
st.set_page_config(page_title="Clasificador Perros vs Gatos", layout="centered")
st.title("Modelo Predictivo Perros vs Gatos - Edwin Guzmán - 2026")
st.write("Suba una imagen para clasificarla mediante la red neuronal MobileNetV2 adaptada.")

IMG_SIZE = (224, 224)

# --- CONFIGURACIÓN DE RUTAS LOCALES Y GITHUB ---
MODEL_DIR = Path("modelo_perros_gatos_mobilenet")
CLASS_PATH = MODEL_DIR / "class_names.json"
# Lista de formatos para intentar cargar de forma dinámica
MODEL_PATHS = [MODEL_DIR / "perros_gatos_net.keras", MODEL_DIR / "perros_gatos_net.h5"]
# -----------------------------------------------

# Diccionario de traducción limpio para la interfaz
LABELS_ES = {
    "Gatos": "Gato",
    "Perros": "Perro"
}

@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                continue
    st.error(f"No se pudo cargar el modelo. Verifique que los archivos estén en '{MODEL_DIR}' dentro de su repositorio de GitHub.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        try:
            with open(CLASS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Si es un diccionario de Keras (ej: {"Gatos":0}), extrae las llaves ordenadas
                if isinstance(data, dict):
                    return sorted(list(data.keys()))
                return data
        except Exception:
            pass
    # Respaldo seguro si el archivo JSON falla o no se lee correctamente
    return ["Gatos", "Perros"]

# Preparar la imagen usando el pipeline nativo de MobileNetV2
def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Importante: Usar el mismo preprocesamiento (escala 1/255 o el nativo de MobileNetV2)
    # Como usamos rescale=1./255 en el ImageDataGenerator, dividimos por 255.0
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    
    # Manejar salidas tanto para clasificación binaria como categórica de 2 neuronas
    if len(preds) == 1:
        # Si la salida es una sola neurona sigmoide
        prob_perro = float(preds[0])
        prob_gato = 1.0 - prob_perro
        probabilidades = [("Gato", prob_gato * 100), ("Perro", prob_perro * 100)]
    else:
        # Si la salida es Softmax (2 neuronas) como configuramos en tu arquitectura
        probabilidades = [
            (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
            for i in range(len(clases))
        ]
    
    # Ordenar de mayor a menor probabilidad
    probabilidades.sort(key=lambda x: x[1], reverse=True)
    return probabilidades

# Cargar los componentes del modelo
modelo = cargar_modelo()
clases = cargar_clases()

# Componente de subida de archivos
archivo = st.file_uploader("Seleccione una imagen de un perro o gato", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Analizando imagen..."):
        resultados = predecir(imagen)
        
    st.subheader("Resultado del Análisis")
    st.success(f"Predicción Principal: **{resultados[0][0]}** ({resultados[0][1]:.2f}%)")

    # Mostrar métricas ordenadas
    st.write("Distribución de confianza:")
    for clase, prob in resultados:
        st.write(f"- **{clase}**: {prob:.2f}%")
else:
    st.info("Cargue una imagen para iniciar la clasificación.")
