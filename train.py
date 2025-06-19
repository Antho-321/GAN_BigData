# train.py
# -*- coding: utf-8 -*-
"""
train.py

Script principal para entrenar la GAN con la arquitectura SAGAN.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Importar desde nuestros módulos locales
import config
from data_loader import load_and_preprocess_mnist
# Importar únicamente los modelos SAGAN
from model import build_sagan_generator, build_sagan_discriminator
from utils import (sample_and_save_images, save_comparison_images,
                   scale_images, calculate_fid)


def train():
    """Función principal para configurar y ejecutar el bucle de entrenamiento de SAGAN."""

    # --- 1. Cargar y Preparar Datos ---
    # SAGAN requiere normalización en [-1, 1] debido a su capa de salida tanh.
    print("Cargando datos con normalización en el rango: [-1, 1]")
    X_train = load_and_preprocess_mnist(normalization_range='-1_to_1')
    
    # SAGAN espera una dimensión de canal explícita (p.ej., 28x28x1).
    X_train = np.expand_dims(X_train, axis=-1)
        
    # --- 2. Construir Modelos y Optimizador ---
    print("Construyendo la arquitectura SAGAN...")
    generator = build_sagan_generator(config.LATENT_DIM)
    discriminator = build_sagan_discriminator(config.IMG_SHAPE)
    optimizer = Adam(learning_rate=config.LEARNING_RATE, beta_1=config.BETA_1)
    
    # SAGAN usa logits, por lo que la función de pérdida debe configurarse así.
    loss_function = BinaryCrossentropy(from_logits=True)

    # --- 3. Compilar Modelos ---
    discriminator.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    discriminator.trainable = False
    gan_model = Sequential([generator, discriminator], name="SAGAN_model")
    gan_model.compile(loss=loss_function, optimizer=optimizer)

    # Imprimir resúmenes para verificación
    print("\n" + "="*50)
    print("Resumen del Generador SAGAN:")
    generator.summary()
    print("\nResumen del Discriminador SAGAN:")
    discriminator.summary()
    print("="*50 + "\n")

    # --- 4. Preparar para el Bucle de Entrenamiento ---
    real_labels = np.ones((config.BATCH_SIZE, 1))
    fake_labels = np.zeros((config.BATCH_SIZE, 1))

    print("Cargando modelo InceptionV3 para cálculo de FID...")
    inception_model = tf.keras.applications.InceptionV3(
        include_top=False, pooling='avg', input_shape=(75, 75, 3)
    )
    print("Modelo InceptionV3 cargado.")

    # --- 5. Bucle de Entrenamiento ---
    print("\nIniciando entrenamiento...")
    for epoch in range(config.EPOCHS):
        # Entrenar el Discriminador
        idx = np.random.randint(0, X_train.shape[0], config.BATCH_SIZE)
        real_imgs = X_train[idx]
        noise = np.random.normal(0, 1, (config.BATCH_SIZE, config.LATENT_DIM))
        fake_imgs = generator.predict(noise, verbose=0)
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Entrenar el Generador
        noise = np.random.normal(0, 1, (config.BATCH_SIZE, config.LATENT_DIM))
        g_loss = gan_model.train_on_batch(noise, real_labels)

        # Mostrar progreso y guardar muestras
        if epoch % 100 == 0:
            print(f"Época {epoch:05d} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        if epoch % config.SAMPLE_INTERVAL == 0:
            # Las funciones de guardado usarán la des-normalización para [-1, 1]
            sample_and_save_images(epoch, generator, config.LATENT_DIM, config.IMAGE_PATH, input_range='-1_to_1')
            save_comparison_images(epoch, generator, config.LATENT_DIM, X_train, config.COMPARISON_PATH, input_range='-1_to_1')

            # Calcular FID
            idx_fid = np.random.randint(0, X_train.shape[0], config.FID_SAMPLES)
            real_images_fid = X_train[idx_fid]
            noise_fid = np.random.normal(0, 1, (config.FID_SAMPLES, config.LATENT_DIM))
            fake_images_fid = generator.predict(noise_fid, verbose=0)
            
            # Los datos ya están en [-1, 1], listos para InceptionV3
            real_images_scaled = scale_images(real_images_fid, (75, 75))
            fake_images_scaled = scale_images(fake_images_fid, (75, 75))
            
            fid_score = calculate_fid(inception_model, real_images_scaled, fake_images_scaled)
            print(f"--- Época {epoch:05d}: FID Score = {fid_score:.3f} ---")


if __name__ == '__main__':
    # Crear directorios para guardar las imágenes
    print(f"Las imágenes de muestra se guardarán en: {config.IMAGE_PATH}")
    print(f"Las comparaciones se guardarán en: {config.COMPARISON_PATH}")
    os.makedirs(config.IMAGE_PATH, exist_ok=True)
    os.makedirs(config.COMPARISON_PATH, exist_ok=True)
    
    # Iniciar el proceso de entrenamiento
    train()