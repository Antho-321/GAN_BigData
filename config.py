# config.py

import numpy as np
import tensorflow as tf

# --- Configuración del Modelo y Entrenamiento ---
LATENT_DIM = 100
IMG_SHAPE = (28, 28, 1)
EPOCHS = 2000
BATCH_SIZE = 64
SAVE_INTERVAL = 200
LEARNING_RATE = 0.0002 # Adam optimizer learning rate
ADAM_BETA_1 = 0.5    # Adam optimizer beta1

# --- Configuración para la Métrica FID ---
# El input_shape mínimo para InceptionV3 es 75x75
INCEPTION_INPUT_SHAPE = (75, 75, 3)
FID_BATCH_SIZE = 512 # Usar un batch más grande para un cálculo de FID más estable

# --- Rutas y Directorios ---
GENERATED_IMAGES_DIR = "/content/drive/MyDrive/Colab Notebooks/generated_images"
EVALUATION_DIR = "/content/drive/MyDrive/Colab Notebooks/evaluacion" # Puedes cambiar esta ruta
CACHE_DIR            = "/content/drive/MyDrive/Colab Notebooks/cache_fid"

# --- Semillas para reproducibilidad ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)