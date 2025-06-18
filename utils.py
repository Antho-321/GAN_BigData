# utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import preprocess_input

def scale_and_convert_to_rgb(images, inception_input_shape):
    """Redimensiona imágenes y las convierte a 3 canales (RGB) para InceptionV3."""
    images_rescaled = (images + 1) * 127.5
    images_resized = tf.image.resize(images_rescaled, (inception_input_shape[0], inception_input_shape[1]), method='nearest')
    images_rgb = tf.image.grayscale_to_rgb(images_resized)
    return images_rgb

def calculate_fid(model, images1, images2):
    """Calcula el Fréchet Inception Distance (FID) entre dos grupos de imágenes."""
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    act1 = model.predict(images1, verbose=0)
    act2 = model.predict(images2, verbose=0)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def sample_images(epoch, generator, latent_dim, save_dir, grid=4):
    """Genera y guarda una grilla de imágenes de muestra."""
    noise = np.random.normal(0, 1, (grid * grid, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(grid, grid, figsize=(5, 5))
    count = 0
    for i in range(grid):
        for j in range(grid):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    
    plt.suptitle(f"Imágenes generadas - Época {epoch}")
    fig.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
    plt.close(fig)

def guardar_imagenes_evaluacion(X_train, generator, latent_dim, epochs, eval_dir, num_imagenes=10):
    """Guarda imágenes reales y generadas para una evaluación visual final."""
    os.makedirs(eval_dir, exist_ok=True)
    print("\n" + "="*50 + "\nINICIANDO FASE DE EVALUACIÓN FINAL\n" + "="*50)
    
    # Guardar imágenes reales
    idx = np.random.randint(0, X_train.shape[0], num_imagenes)
    real_imgs = 0.5 * X_train[idx] + 0.5
    fig, axs = plt.subplots(2, 5, figsize=(8, 4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(real_imgs[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.suptitle("10 Muestras Reales del Dataset")
    fig.savefig(os.path.join(eval_dir, "muestras_reales.png"))
    plt.close(fig)
    print(f"Imágenes reales guardadas en '{os.path.join(eval_dir, 'muestras_reales.png')}'")

    # Guardar imágenes generadas
    noise = np.random.normal(0, 1, (num_imagenes, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(2, 5, figsize=(8, 4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(gen_imgs[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.suptitle(f"10 Imágenes Generadas (Modelo Final - Época {epochs})")
    fig.savefig(os.path.join(eval_dir, f"imagenes_generadas_final_epoch_{epochs}.png"))
    plt.close(fig)
    print(f"Imágenes generadas guardadas en '{os.path.join(eval_dir, f'imagenes_generadas_final_epoch_{epochs}.png')}'")
    print("\n" + "="*50)