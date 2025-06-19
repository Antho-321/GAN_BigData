# -*- coding: utf-8 -*-
"""
utils.py

Funciones auxiliares para el entrenamiento de la GAN, como el cálculo
de métricas (FID) y la generación de imágenes de muestra.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from skimage.transform import resize
import os

def scale_images(images, new_shape):
    """Redimensiona imágenes a un nuevo tamaño y las convierte a formato RGB."""
    images_list = []
    for image in images:
        new_image = resize(image, new_shape, 0)
        new_image_rgb = np.stack([new_image] * 3, axis=-1)
        images_list.append(new_image_rgb)
    return np.asarray(images_list)

def calculate_fid(model, images1, images2):
    """Calcula el Fréchet Inception Distance (FID) entre dos grupos de imágenes."""
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

def sample_and_save_images(epoch, generator, latent_dim, save_path, input_range='0_to_1'):
    """
    Genera y guarda una grilla de imágenes de muestra.

    Args:
        epoch (int): La época actual, para nombrar el archivo.
        generator (tf.keras.Model): El modelo generador entrenado.
        latent_dim (int): La dimensionalidad del espacio latente.
        save_path (str): El directorio donde se guardarán las imágenes.
        input_range (str): El rango de normalización de las imágenes ('0_to_1' o '-1_to_1').
    """
    r, c = 5, 5  # Filas y columnas para la grilla de imágenes
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)

    # Des-normalizar según el rango de entrada para poder visualizar
    if input_range == '0_to_1':
        gen_imgs = (gen_imgs * 255).astype(np.uint8)
    elif input_range == '-1_to_1':
        gen_imgs = (gen_imgs * 127.5 + 127.5).astype(np.uint8)

    # Crear la figura y los ejes para la grilla
    fig, axs = plt.subplots(r, c, figsize=(10, 10))
    
    # Llenar la grilla con las imágenes generadas
    cnt = 0
    for i in range(r):
        for j in range(c):
            # Las imágenes generadas pueden tener un canal al final (28, 28, 1)
            # imshow lo maneja bien, pero si no, se puede usar np.squeeze()
            if gen_imgs[cnt].shape[-1] == 1:
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            else:
                axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
            
    # Asegurarse de que el directorio de destino exista
    os.makedirs(save_path, exist_ok=True)
    
    # Guardar la figura en un archivo
    fig.savefig(os.path.join(save_path, f"mnist_{epoch:05d}.png"))
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)

def save_comparison_images(epoch, generator, latent_dim, real_dataset, save_path, num_samples=5, input_range='0_to_1'):
    """
    Guarda una comparativa de imágenes reales y generadas.

    Args:
        epoch (int): La época actual.
        generator (tf.keras.Model): El modelo generador.
        latent_dim (int): La dimensionalidad del espacio latente.
        real_dataset (np.array): El dataset real para tomar muestras.
        save_path (str): El directorio donde se guardará la imagen de comparación.
        num_samples (int): El número de imágenes a comparar.
        input_range (str): El rango de normalización de las imágenes.
    """
    # 1. Preparar las imágenes
    # Seleccionar N imágenes reales aleatorias del dataset
    idx = np.random.randint(0, real_dataset.shape[0], num_samples)
    real_imgs_normalized = real_dataset[idx]
    
    # Generar N imágenes falsas
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    fake_imgs_normalized = generator.predict(noise, verbose=0)

    # 2. Des-normalizar ambas para visualización
    if input_range == '0_to_1':
        real_imgs = (real_imgs_normalized * 255).astype(np.uint8)
        fake_imgs = (fake_imgs_normalized * 255).astype(np.uint8)
    elif input_range == '-1_to_1':
        real_imgs = (real_imgs_normalized * 127.5 + 127.5).astype(np.uint8)
        fake_imgs = (fake_imgs_normalized * 127.5 + 127.5).astype(np.uint8)

    # 3. Crear la figura y los ejes
    fig, axs = plt.subplots(2, num_samples, figsize=(12, 5))
    fig.suptitle(f'Época: {epoch}', fontsize=16)

    for i in range(num_samples):
        # Función auxiliar para manejar la forma de la imagen
        def get_image_to_show(img_array):
            if img_array.shape[-1] == 1:
                return img_array[:, :, 0]
            return img_array

        # Mostrar imágenes originales en la primera fila
        axs[0, i].imshow(get_image_to_show(real_imgs[i]), cmap='gray')
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')

        # Mostrar imágenes generadas en la segunda fila
        axs[1, i].imshow(get_image_to_show(fake_imgs[i]), cmap='gray')
        axs[1, i].set_title("Generada")
        axs[1, i].axis('off')

    # 4. Guardar la figura
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"compare_{epoch:05d}.png"))
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)