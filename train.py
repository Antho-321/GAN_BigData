# train.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam

# Importar desde nuestros módulos locales
import config
from model import build_generator, build_discriminator, build_gan
from utils import sample_images, calculate_fid, scale_and_convert_to_rgb, guardar_imagenes_evaluacion

def main():
    """Función principal para ejecutar el entrenamiento de la GAN."""
    
    # --- 1. Carga y Preparación de Datos ---
    print("Cargando y preparando datos de MNIST...")
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)
    
    # Crear carpeta para imágenes generadas
    os.makedirs(config.GENERATED_IMAGES_DIR, exist_ok=True)

    # --- 2. Construcción de Modelos ---
    print("Construyendo los modelos: Generador, Discriminador y GAN...")
    generator = build_generator(config.LATENT_DIM)
    discriminator = build_discriminator(config.IMG_SHAPE)
    gan = build_gan(generator, discriminator, config.LATENT_DIM)

    # --- 3. Compilación de Modelos ---
    discriminator_optimizer = Adam(learning_rate=config.LEARNING_RATE, beta_1=config.ADAM_BETA_1)
    gan_optimizer = Adam(learning_rate=config.LEARNING_RATE, beta_1=config.ADAM_BETA_1)
    
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    # Cargar modelo para cálculo de FID
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=config.INCEPTION_INPUT_SHAPE)

    print("Imágenes iniciales guardadas. Comenzando entrenamiento…")
    valid = np.ones((config.BATCH_SIZE, 1))
    fake = np.zeros((config.BATCH_SIZE, 1))

    for epoch in range(config.EPOCHS + 1):
        # --- Entrenar Discriminador ---
        idx = np.random.randint(0, X_train.shape[0], config.BATCH_SIZE)
        real_imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (config.BATCH_SIZE, config.LATENT_DIM))
        gen_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # --- Entrenar Generador ---
        noise = np.random.normal(0, 1, (config.BATCH_SIZE, config.LATENT_DIM))
        g_loss = gan.train_on_batch(noise, valid)

        # --- Reporte de Progreso y Evaluación (FID) ---
        if epoch % config.SAVE_INTERVAL == 0:
            print(f"[{epoch}] D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}% | G loss: {g_loss:.4f}", end="")
            
            sample_images(epoch, generator, config.LATENT_DIM, config.GENERATED_IMAGES_DIR)

            # Preparar imágenes para FID
            idx_fid = np.random.randint(0, X_train.shape[0], config.FID_BATCH_SIZE)
            real_fid_imgs = X_train[idx_fid]
            noise_fid = np.random.normal(0, 1, (config.FID_BATCH_SIZE, config.LATENT_DIM))
            gen_fid_imgs = generator.predict(noise_fid, verbose=0)
            
            real_fid_imgs_rgb = scale_and_convert_to_rgb(real_fid_imgs, config.INCEPTION_INPUT_SHAPE)
            gen_fid_imgs_rgb = scale_and_convert_to_rgb(gen_fid_imgs, config.INCEPTION_INPUT_SHAPE)

            # Calcular y mostrar FID
            fid_score = calculate_fid(inception_model, real_fid_imgs_rgb, gen_fid_imgs_rgb)
            print(f" | FID: {fid_score:.2f}")

    # --- 5. Evaluación Final ---
    guardar_imagenes_evaluacion(
        generator=generator,
        latent_dim=config.LATENT_DIM,
        epochs=config.EPOCHS,
        eval_dir=config.EVALUATION_DIR
    )
    
    print("\nProceso de entrenamiento y evaluación completado.")

if __name__ == '__main__':
    main()