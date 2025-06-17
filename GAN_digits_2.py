# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 08:55:07 2025

@author: IVAN
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
np.random.seed(42)
tf.random.set_seed(42)

# Hiperparámetros
latent_dim = 200  # Dimensión del ruido de entrada
img_shape = (28, 28)  # Forma de las imágenes MNIST
epochs = 800  # Número de épocas (aumentado)
batch_size = 64
lr = 0.0001  # Learning rate
beta_1 = 0.5  # Parámetro para Adam

# Cargar y preprocesar datos MNIST
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train / 255  # Normalizar imágenes en [0, 1]
X_train = np.expand_dims(X_train, axis=3)  # Añadir dimensión de canal (28, 28, 1)



# Seleccionar algunas imágenes aleatorias (ej. 9, 16) de ejemplo
num_images = 16
random_indices = np.random.randint(0, X_train.shape[0], num_images)
sample_images = X_train[random_indices] / 255.0
sample_labels = y_train[random_indices]

# Configurar el grid de visualización
rows = 4
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

# Mostrar las imágenes
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')  # 'gray' para escala de grises
    ax.set_title(f"Label: {sample_labels[i]}", fontsize=10)
    ax.axis('off')  # Ocultar ejes

plt.suptitle('Ejemplos del Dataset MNIST', fontsize=14)
plt.tight_layout()
plt.show()



# ======================================
# Generador (Mejorado)
# ======================================
def build_generator():
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(np.prod(img_shape), activation='sigmoid'),  # Salida en [0, 1]
        Reshape(img_shape)
    ])
    return model

# ======================================
# Discriminador (Mejorado)
# ======================================
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=img_shape),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')  # Clasifica real (1) vs. falso (0)
    ])
    return model

# Compilar el discriminador
discriminator = build_discriminator()
discriminator.compile(
    optimizer=Adam(learning_rate=lr, beta_1=beta_1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Compilar el generador
generator = build_generator()

# Congelar el discriminador durante el entrenamiento del generador
discriminator.trainable = False

# Modelo GAN (Generador -> Discriminador)
gan = Sequential([generator, discriminator])
gan.compile(
    optimizer=Adam(learning_rate=lr, beta_1=beta_1),
    loss='binary_crossentropy'
)

# ======================================
# Entrenamiento (con visualización)
# ======================================
def train_gan(epochs, batch_size, sample_interval=50):
    # Etiquetas para datos reales y falsos
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # ---------------------
        # Entrenar Discriminador
        # ---------------------
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # -----------------
        # Entrenar Generador
        # -----------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)  # Engañar al discriminador
        
        # Progreso
        if epoch % sample_interval == 0:
            print(f"Época {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
            sample_images(epoch)

# Guardar imágenes generadas cada cierto tiempo
# def sample_images(epoch):
#     r, c = 4, 4
#     noise = np.random.normal(0, 1, (r * c, latent_dim))
#     gen_imgs = generator.predict(noise)  # Forma: (16, 28, 28)
    
#     # Re-escalar imágenes de [-1, 1] a [0, 1]
#     gen_imgs = 0.5 * gen_imgs + 0.5
    
#     fig, axs = plt.subplots(r, c)
#     cnt = 0
#     for i in range(r):
#         for j in range(c):
#             axs[i,j].imshow(gen_imgs[cnt], cmap='gray')  # ¡Corregido! Sin [:, :, 0]
#             axs[i,j].axis('off')
#             cnt += 1
#     fig.savefig(f"mnist_gan_epoch_{epoch}.png")
#     plt.close()

# Un cuadrícula de 4x4 (16 imágenes en total) generadas por el modelo en esa época.
def sample_images(epoch):
    r, c = 4, 4  # Grid de 4x4 imágenes
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    
    # Re-escalar imágenes de [-1, 1] a [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    
    plt.suptitle(f'Época {epoch}', fontsize=12)
    plt.tight_layout()
    plt.show()  # Muestra la figura en pantalla
    plt.close()  # Cierra la figura para liberar memoria

# Iniciar entrenamiento
train_gan(epochs=epochs, batch_size=batch_size, sample_interval=100)