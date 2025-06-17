# -*- coding: utf-8 -*-
"""
Script para entrenar una GAN utilizando las arquitecturas definidas en model.py.

Este script importa los modelos, carga el dataset MNIST, y ejecuta el
proceso de entrenamiento para generar dígitos escritos a mano.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from skimage.transform import resize

# -----------------------------------------------------------------------------
# Paso 1: Importar las arquitecturas desde el archivo model.py
# -----------------------------------------------------------------------------
# Asegúrate de que 'model.py' esté en el mismo directorio.
# Nota: La siguiente línea asumirá que 'model.py' existe. Si no, se producirá un error.
from model import (build_simple_generator, build_simple_discriminator,
                   build_improved_generator, build_improved_discriminator)

# -----------------------------------------------------------------------------
# Paso 2: Configuración y Hiperparámetros
# -----------------------------------------------------------------------------
# Elige qué arquitectura usar. True para la mejorada, False para la simple.
USE_IMPROVED_MODEL = True # Se usará la implementación de ejemplo de arriba

# Hiperparámetros comunes
EPOCHS = 20000
BATCH_SIZE = 128
SAMPLE_INTERVAL = 500  # Frecuencia en épocas para guardar imágenes de muestra

# Crear directorios para guardar las imágenes generadas y de comparación
os.makedirs("/content/drive/MyDrive/colab/images", exist_ok=True)
os.makedirs("/content/drive/MyDrive/colab/comparison", exist_ok=True)  # <-- AÑADIDO

# -----------------------------------------------------------------------------
# Paso 3: Cargar y Preprocesar el Dataset (MNIST)
# -----------------------------------------------------------------------------
(X_train, _), (_, _) = mnist.load_data()

# Normalizar las imágenes en el rango [0, 1]
X_train = X_train / 255.0

# -----------------------------------------------------------------------------
# Paso 4: Construir y Compilar los Modelos
# -----------------------------------------------------------------------------
if USE_IMPROVED_MODEL:
    print("Usando la arquitectura MEJORADA de la GAN.")
    latent_dim = 200
    img_shape = (28, 28)
    generator = build_improved_generator(latent_dim=latent_dim, img_shape=img_shape)
    discriminator = build_improved_discriminator(img_shape=img_shape)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5) # Optimizador recomendado para GANs
else:
    print("Usando la arquitectura SIMPLE de la GAN.")
    latent_dim = 150
    img_shape = (28, 28)
    generator = build_simple_generator(latent_dim=latent_dim)
    discriminator = build_simple_discriminator(img_shape=img_shape)
    optimizer = Adam(learning_rate=0.0002)

# --- Compilar el Discriminador ---
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# --- Compilar el modelo GAN combinado ---
# Al entrenar el generador, los pesos del discriminador se congelan.
discriminator.trainable = False

# El modelo GAN apila el generador y el discriminador
gan_model = Sequential([generator, discriminator])
gan_model.compile(loss='binary_crossentropy', optimizer=optimizer)

print("\nResumen del Generador:")
generator.summary()
print("\nResumen del Discriminador:")
discriminator.summary()

# -----------------------------------------------------------------------------
# Paso 5: Definir Funciones Auxiliares y el Bucle de Entrenamiento
# -----------------------------------------------------------------------------
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))

def scale_images(images, new_shape):
    """
    Redimensiona imágenes en escala de grises (28x28) a un nuevo tamaño
    y las convierte a formato RGB (3 canales).
    """
    images_list = []
    for image in images:
        # Redimensiona la imagen
        new_image = resize(image, new_shape, 0)
        # Apila 3 veces el canal único para simular RGB
        new_image_rgb = np.stack([new_image] * 3, axis=-1)
        images_list.append(new_image_rgb)
    return np.asarray(images_list)

def calculate_fid(model, images1, images2):
    """
    Calcula el Fréchet Inception Distance (FID) entre dos grupos de imágenes.
    """
    # Calcula las activaciones (características) usando el modelo InceptionV3
    act1 = model.predict(images1, verbose=0)
    act2 = model.predict(images2, verbose=0)
    
    # Calcula la media y la matriz de covarianza para cada grupo
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Calcula la suma de las diferencias cuadradas entre las medias
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calcula la raíz cuadrada del producto de las covarianzas
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Corrige si el resultado es un número complejo
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    # Calcula el puntaje FID final
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def sample_and_save_images(epoch, generator, latent_dim):
    """Genera imágenes de muestra y las guarda en un archivo."""
    r, c = 5, 5  # 5x5 grid
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)

    # Re-escalar imágenes de [0, 1] a [0, 255]
    gen_imgs = (gen_imgs * 255).astype(np.uint8)

    fig, axs = plt.subplots(r, c, figsize=(10, 10))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"/content/drive/MyDrive/colab/images/mnist_{epoch:05d}.png")
    plt.close()

# <-- FUNCIÓN AÑADIDA -->
def save_comparison_images(epoch, generator, latent_dim, real_dataset, num_samples=5):
    """
    Guarda una comparativa de imágenes reales del dataset y imágenes generadas.
    """
    # --- 1. Preparar las imágenes ---
    # Seleccionar N imágenes reales aleatorias del dataset
    idx = np.random.randint(0, real_dataset.shape[0], num_samples)
    real_imgs = real_dataset[idx]

    # Generar N imágenes falsas
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    fake_imgs = generator.predict(noise, verbose=0)

    # --- 2. Crear la figura y guardar ---
    fig, axs = plt.subplots(2, num_samples, figsize=(12, 5))
    fig.suptitle(f'Época: {epoch}', fontsize=16)

    for i in range(num_samples):
        # Mostrar imágenes originales en la primera fila
        axs[0, i].imshow(real_imgs[i], cmap='gray')
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')

        # Mostrar imágenes generadas en la segunda fila
        axs[1, i].imshow(fake_imgs[i], cmap='gray')
        axs[1, i].set_title("Generada")
        axs[1, i].axis('off')

    # Guardar la figura en el directorio 'comparison'
    fig.savefig(f"/content/drive/MyDrive/colab/comparison/compare_{epoch:05d}.png")
    plt.close(fig)


def train():
    """Función principal para ejecutar el bucle de entrenamiento."""
    real_labels = np.ones((BATCH_SIZE, 1))
    fake_labels = np.zeros((BATCH_SIZE, 1))
    
    # Número de imágenes para un cálculo de FID estable
    FID_SAMPLES = 1000

    for epoch in range(EPOCHS):
        # -------------------------
        #  Entrenar el Discriminador
        # -------------------------
        idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        real_imgs = X_train[idx]
        noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Entrenar el Generador
        # ---------------------
        noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
        g_loss = gan_model.train_on_batch(noise, real_labels)

        # ------------------------------------------
        #  Mostrar Progreso, Guardar Muestras y Calcular FID
        # ------------------------------------------
        if epoch % 100 == 0:
            print(f"Época {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        # El cálculo del FID es costoso, se hace periódicamente
        if epoch % SAMPLE_INTERVAL == 0:
            sample_and_save_images(epoch, generator, latent_dim)
            
            # <-- LLAMADA A LA FUNCIÓN AÑADIDA -->
            save_comparison_images(epoch, generator, latent_dim, X_train, num_samples=5)

            # 1. Preparar imágenes reales
            idx_fid = np.random.randint(0, X_train.shape[0], FID_SAMPLES)
            real_images_fid = X_train[idx_fid]
            real_images_fid_scaled = scale_images(real_images_fid, (75, 75))
            real_images_fid_prepared = preprocess_input(real_images_fid_scaled)
            
            # 2. Preparar imágenes falsas
            noise_fid = np.random.normal(0, 1, (FID_SAMPLES, latent_dim))
            fake_images_fid = generator.predict(noise_fid, verbose=0)
            fake_images_fid_scaled = scale_images(fake_images_fid, (75, 75))
            fake_images_fid_prepared = preprocess_input(fake_images_fid_scaled)
            
            # 3. Calcular FID
            fid_score = calculate_fid(inception_model, real_images_fid_prepared, fake_images_fid_prepared)
            print(f"--- Época {epoch}: FID Score = {fid_score:.3f} ---")

# -----------------------------------------------------------------------------
# Paso 6: Iniciar el Entrenamiento
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    train()