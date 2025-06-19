# utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import tensorflow_probability as tfp      # ⬅️  1)  IMPORTANTE

def scale_and_convert_to_rgb(images, inception_input_shape):
    """
    - images deben estar en el rango [-1, 1].
    - Devuelve un tensor (batch, H, W, 3) adecuado para Inception V3.
    """
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    # ↳ Si vienen como (batch, h, w), añadimos canal = 1
    if images.shape.rank == 3:                 # (B, H, W)
        images = images[..., tf.newaxis]       # → (B, H, W, 1)

    # Re-escala de [-1,1] → [0,255]
    images_rescaled = (images + 1.0) * 127.5

    # Redimensiona al tamaño mínimo de Inception (75×75 aquí)
    target_h, target_w = inception_input_shape[:2]
    images_resized = tf.image.resize(images_rescaled,
                                     (target_h, target_w),
                                     method='bilinear')

    # Convierte 1 canal → 3 canales. Si ya son 3, no hace nada.
    if images_resized.shape[-1] == 1:
        images_rgb = tf.image.grayscale_to_rgb(images_resized)
    else:
        images_rgb = images_resized

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

def guardar_imagenes_evaluacion(generator, latent_dim, epochs, eval_dir, num_imagenes=16):
    """Guarda imágenes generadas para una evaluación visual final."""
    os.makedirs(eval_dir, exist_ok=True)
    print("\n" + "="*50 + "\nINICIANDO FASE DE EVALUACIÓN FINAL\n" + "="*50)

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

def batch_fid(mu_real, sigma_real, acts_fake):
    # 1) Usa float16 para reducir RAM **solo** en las activaciones
    acts_fake_f16 = tf.cast(acts_fake, tf.float16)
    mu_fake_f16   = tf.reduce_mean(acts_fake_f16, axis=0)
    diff_mu_f16   = mu_fake_f16 - tf.cast(mu_real, tf.float16)

    # 2) Covarianza también puede calcularse en f16
    cov_fake_f16  = tf.cast(tfp.stats.covariance(acts_fake_f16), tf.float16)

    # 3) Pero sqrtm requiere f32 → convierte justo antes de llamarla
    cov_fake_f32  = tf.cast(cov_fake_f16,  tf.float32)
    sigma_real_f32= tf.cast(sigma_real,    tf.float32)

    # Mover a CPU si quieres liberar VRAM
    with tf.device("/CPU:0"):
        cov_mean_f32 = tf.linalg.sqrtm(tf.matmul(sigma_real_f32, cov_fake_f32))

    cov_mean_f32 = tf.math.real(cov_mean_f32)

    # 4) Vuelve a f16 para acabar la fórmula (opcional)
    cov_mean_f16 = tf.cast(cov_mean_f32, tf.float16)

    fid_f16 = tf.reduce_sum(tf.square(diff_mu_f16)) + tf.linalg.trace(
                tf.cast(sigma_real_f16, tf.float16) +
                cov_fake_f16 - 2.0 * cov_mean_f16)

    # Devuelve en float32 por claridad
    return tf.cast(fid_f16, tf.float32)