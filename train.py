# train.py
# -*- coding: utf-8 -*-
"""
train.py

Script principal para entrenar la GAN con la arquitectura SAGAN, utilizando
las mejores prácticas para estabilidad y rendimiento:
- Hinge Loss (pérdida de bisagra)
- Dos optimizadores Adam con betas específicos (β1=0.0, β2=0.9)
- Label smoothing para métricas
- Bucle de entrenamiento personalizado con @tf.function para alta eficiencia
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Importar desde nuestros módulos locales
import config
from data_loader import load_and_preprocess_mnist
from model import build_sagan_generator, build_sagan_discriminator
from utils import (sample_and_save_images, save_comparison_images,
                   scale_images, calculate_fid)


# --- 1. Funciones de Pérdida (Hinge Loss) ---
# Implementación de Hinge Loss, que funciona mejor para SAGAN que BinaryCrossentropy.
def d_loss_hinge(real_logits, fake_logits):
    """Pérdida del discriminador (Hinge Loss)."""
    loss_real = tf.reduce_mean(tf.nn.relu(1. - real_logits))
    loss_fake = tf.reduce_mean(tf.nn.relu(1. + fake_logits))
    return loss_real + loss_fake

def g_loss_hinge(fake_logits):
    """Pérdida del generador (Hinge Loss)."""
    return -tf.reduce_mean(fake_logits)


# --- 2. Clase Entrenadora con Bucle Personalizado ---
# Encapsula los modelos, optimizadores y el paso de entrenamiento en una clase.
class SaganTrainer:
    """Gestiona el proceso de entrenamiento de la SAGAN."""
    def __init__(self):
        # Construir Modelos
        self.G = build_sagan_generator(config.LATENT_DIM)
        self.D = build_sagan_discriminator(config.IMG_SHAPE)

        # Optimizadores con parámetros del paper de SAGAN
        self.g_opt = Adam(config.LEARNING_RATE, beta_1=0.0, beta_2=0.9)
        self.d_opt = Adam(config.LEARNING_RATE, beta_1=0.0, beta_2=0.9)

        # Métrica para monitorizar el accuracy del discriminador (solo como referencia)
        self.d_accuracy = tf.keras.metrics.BinaryAccuracy()
        
        # Imprimir resúmenes para verificación
        print("\n" + "="*50)
        print("Resumen del Generador SAGAN:")
        self.G.summary()
        print("\nResumen del Discriminador SAGAN:")
        self.D.summary()
        print("="*50 + "\n")

    @tf.function
    def train_step(self, real_imgs):
        """
        Ejecuta un único paso de entrenamiento.
        Compilado con @tf.function para un rendimiento óptimo.
        """
        batch_size = tf.shape(real_imgs)[0]
        noise = tf.random.normal([batch_size, config.LATENT_DIM])

        # --- Paso de entrenamiento del Discriminador ---
        with tf.GradientTape() as tape_d:
            fake_imgs = self.G(noise, training=True)
            
            real_logits = self.D(real_imgs, training=True)
            fake_logits = self.D(fake_imgs, training=True)
            
            d_loss = d_loss_hinge(real_logits, fake_logits)
        
        # Calcular y aplicar gradientes
        grads_d = tape_d.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grads_d, self.D.trainable_variables))

        # Actualizar métrica de accuracy (con label smoothing para estabilizar)
        real_labels = tf.fill([batch_size, 1], 0.9) # Label smoothing
        fake_labels = tf.zeros([batch_size, 1])
        all_logits = tf.concat([real_logits, fake_logits], axis=0)
        all_labels = tf.concat([real_labels, fake_labels], axis=0)
        self.d_accuracy.update_state(all_labels, tf.sigmoid(all_logits))

        # --- Paso de entrenamiento del Generador ---
        with tf.GradientTape() as tape_g:
            # Generar nuevas imágenes falsas
            noise = tf.random.normal([batch_size, config.LATENT_DIM])
            fake_imgs = self.G(noise, training=True)
            fake_logits = self.D(fake_imgs, training=True)
            
            g_loss = g_loss_hinge(fake_logits)

        # Calcular y aplicar gradientes
        grads_g = tape_g.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grads_g, self.G.trainable_variables))

        return d_loss, g_loss


def train():
    """Función principal para configurar y ejecutar el bucle de entrenamiento."""

    # --- 1. Cargar y Preparar Datos ---
    print("Cargando datos con normalización en el rango: [-1, 1]")
    X_train = load_and_preprocess_mnist(normalization_range='-1_to_1')
    
    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=-1)
    
    # Usar tf.data para un pipeline de datos eficiente
    dataset = (tf.data.Dataset.from_tensor_slices(X_train)
               .shuffle(60_000)
               .batch(config.BATCH_SIZE, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))

    # --- 2. Inicializar Entrenador y Modelo para FID ---
    print("Construyendo la arquitectura SAGAN y optimizadores...")
    trainer = SaganTrainer()

    print("Cargando modelo InceptionV3 para cálculo de FID...")
    inception_model = tf.keras.applications.InceptionV3(
        include_top=False, pooling='avg', input_shape=(75, 75, 3)
    )
    print("Modelo InceptionV3 cargado.")

    # --- 3. Bucle de Entrenamiento Principal ---
    print("\nIniciando entrenamiento...")
    step = 0
    for epoch in range(config.EPOCHS):
        tic = time.time()
        for real_imgs_batch in dataset:
            # Ejecutar un paso de entrenamiento
            d_loss, g_loss = trainer.train_step(real_imgs_batch)
            
            # --- Logging y Guardado de Muestras ---
            if step % 100 == 0:
                acc = trainer.d_accuracy.result().numpy() * 100
                print(f"Época {epoch:03d} St.{step:05d} [D loss: {d_loss:.4f}, acc.: {acc:.2f}%] [G loss: {g_loss:.4f}]")
                trainer.d_accuracy.reset_state()

            if step % config.SAMPLE_INTERVAL == 0:
                # Guardar imágenes de muestra y comparaciones
                sample_and_save_images(step, trainer.G, config.LATENT_DIM, config.IMAGE_PATH, input_range='-1_to_1')
                save_comparison_images(step, trainer.G, config.LATENT_DIM, X_train, config.COMPARISON_PATH, input_range='-1_to_1')
                
                # Calcular FID
                idx_fid = np.random.randint(0, X_train.shape[0], config.FID_SAMPLES)
                real_images_fid = X_train[idx_fid]
                noise_fid = tf.random.normal([config.FID_SAMPLES, config.LATENT_DIM])
                fake_images_fid = trainer.G.predict(noise_fid, verbose=0)
                
                real_images_scaled = scale_images(real_images_fid, (75, 75))
                fake_images_scaled = scale_images(fake_images_fid, (75, 75))
                
                fid_score = calculate_fid(inception_model, real_images_scaled, fake_images_scaled)
                print(f"--- Paso {step:05d}: FID Score = {fid_score:.3f} ---")
            
            step += 1
            
        print(f"⏱️  Época {epoch} terminada en {time.time()-tic:.1f}s")


if __name__ == '__main__':
    # Crear directorios para guardar las imágenes si no existen
    print(f"Las imágenes de muestra se guardarán en: {config.IMAGE_PATH}")
    print(f"Las comparaciones se guardarán en: {config.COMPARISON_PATH}")
    os.makedirs(config.IMAGE_PATH, exist_ok=True)
    os.makedirs(config.COMPARISON_PATH, exist_ok=True)
    
    # Iniciar el proceso de entrenamiento
    train()