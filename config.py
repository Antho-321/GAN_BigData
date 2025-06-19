# config.py
"""
config.py

Hiperparámetros y configuraciones para el entrenamiento del modelo SAGAN.
"""
import os

# --- Hiperparámetros de Entrenamiento ---
EPOCHS = 10000  # SAGAN puede necesitar más épocas para converger
BATCH_SIZE = 64   # Un batch size más pequeño es común en modelos complejos
LEARNING_RATE = 0.0002
BETA_1 = 0.5      # Parámetro recomendado para el optimizador Adam en GANs

# --- Dimensiones y Forma (Específicas para SAGAN) ---
LATENT_DIM = 128
# La forma de la imagen DEBE incluir el canal para el discriminador de SAGAN
IMG_SHAPE = (28, 28, 1)

# --- Configuración de Salidas ---
# Frecuencia (en épocas) para guardar imágenes de muestra y calcular FID
SAMPLE_INTERVAL = 1000
# Número de imágenes para un cálculo de FID estable
FID_SAMPLES = 1000

# --- Rutas de Archivos ---
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/resultados_sagan"
IMAGE_PATH = os.path.join(BASE_PATH, "images")
COMPARISON_PATH = os.path.join(BASE_PATH, "comparison")