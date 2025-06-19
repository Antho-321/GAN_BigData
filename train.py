# --- IMPORTS ---
import os, numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam

import config
from model import build_generator, build_discriminator
from utils import (sample_images, scale_and_convert_to_rgb, batch_fid,
                   guardar_imagenes_evaluacion)

# ---------- MAIN ----------
def main():
    """Función principal para el entrenamiento de la GAN con FID."""
    # --- 1. Carga y Preparación de Datos ---
    print("Cargando y preparando datos de MNIST...")
    (X_train, _), (_,_) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1) # o X_train[..., None]

    # Crear directorios si no existen
    os.makedirs(config.GENERATED_IMAGES_DIR, exist_ok=True)
    os.makedirs(config.EVALUATION_DIR, exist_ok=True)

    # --- 2. Construcción de Modelos ---
    print("Construyendo los modelos: Generador y Discriminador...")
    generator     = build_generator(config.LATENT_DIM)
    discriminator = build_discriminator(config.IMG_SHAPE)

    # --- 3. Optimizadores ---
    disc_opt = Adam(config.LEARNING_RATE, beta_1=config.ADAM_BETA_1)
    gen_opt  = Adam(config.LEARNING_RATE, beta_1=config.ADAM_BETA_1)

    discriminator.compile(optimizer=disc_opt,
                          loss='binary_crossentropy', metrics=['accuracy'])
    # NOTA: El modelo GAN combinado no se compila aquí porque el generador se entrena
    # de forma personalizada con la función `train_gen_step`.

    # --- 4. Inception + Estadísticos Reales para FID ---
    print("Cargando modelo Inception y estadísticos FID precalculados...")
    inception = InceptionV3(include_top=False, pooling='avg',
                            input_shape=config.INCEPTION_INPUT_SHAPE)
    inception.trainable = False
    
    stats = np.load("fid_mnist.npz")
    mu_real    = tf.constant(stats["mu"],    dtype=tf.float32)
    sigma_real = tf.constant(stats["sigma"], dtype=tf.float32)

    # --- Configuración del Bucle de Entrenamiento ---
    λ_FID = 10.0  # Hiperparámetro para ponderar la pérdida FID
    bce   = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @tf.function
    def train_gen_step():
        """Paso de entrenamiento para el generador con pérdida adversaria y FID."""
        noise  = tf.random.normal([config.BATCH_SIZE, config.LATENT_DIM])
        valid  = tf.ones((config.BATCH_SIZE, 1), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # 1. Generar imágenes falsas
            fake_imgs  = generator(noise, training=True)
            # 2. Calcular pérdida adversaria (el generador intenta engañar al discriminador)
            validity   = discriminator(fake_imgs, training=False)
            adv_loss   = bce(valid, validity)
            
            # 3. Calcular pérdida FID
            # Redimensionar y convertir a RGB para InceptionV3
            fid_subset = fake_imgs[:config.FID_BATCH_SIZE]
            fake_rgb   = scale_and_convert_to_rgb(fid_subset, config.INCEPTION_INPUT_SHAPE)
            # Obtener activaciones de Inception para las imágenes generadas
            acts_fake  = inception(fake_rgb, training=False)
            # Calcular FID respecto a los estadísticos de las imágenes reales
            fid_loss   = batch_fid(mu_real, sigma_real, acts_fake)
            
            # 4. Pérdida total del generador
            total_loss = adv_loss + λ_FID * fid_loss
            
        # Calcular y aplicar gradientes
        grads = tape.gradient(total_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(grads, generator.trainable_variables))
        
        return adv_loss, fid_loss, total_loss

    # Etiquetas para el entrenamiento del discriminador
    valid = np.ones((config.BATCH_SIZE, 1))
    fake  = np.zeros((config.BATCH_SIZE, 1))

    print("Comenzando el bucle de entrenamiento...")
    for epoch in range(config.EPOCHS + 1):
        # --- 1) Entrenamiento del Discriminador ---
        # Seleccionar un batch aleatorio de imágenes reales
        idx        = np.random.randint(0, X_train.shape[0], config.BATCH_SIZE)
        real_imgs  = X_train[idx]
        
        # Generar un batch de imágenes falsas
        noise      = np.random.normal(0, 1, (config.BATCH_SIZE, config.LATENT_DIM))
        gen_imgs   = generator.predict(noise, verbose=0)

        # Entrenar el discriminador (primero con imágenes reales, luego con falsas)
        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs,  fake)
        d_loss      = 0.5 * np.add(d_loss_real, d_loss_fake)

        # --- 2) Entrenamiento del Generador (con FID) ---
        # Ejecutar un paso de entrenamiento para el generador
        adv_loss, fid_loss, g_loss = train_gen_step()

        # --- 3) Reporte de Progreso y Muestreo de Imágenes ---
        if epoch % config.SAVE_INTERVAL == 0:
            print(f"[{epoch}] D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%"
                  f" | G adv: {adv_loss:.4f} | FID term: {fid_loss:.2f}"
                  f" | Total G: {g_loss:.4f}")
            # Guardar imágenes de muestra para visualización
            sample_images(epoch, generator, config.LATENT_DIM,
                          config.GENERATED_IMAGES_DIR)

    # --- Evaluación Final ---
    print("\nEntrenamiento completado. Guardando imágenes finales para evaluación...")
    guardar_imagenes_evaluacion(
        generator=generator, 
        latent_dim=config.LATENT_DIM,
        epochs=config.EPOCHS, 
        eval_dir=config.EVALUATION_DIR
    )
    print("Proceso finalizado.")

if __name__ == "__main__":
    main()