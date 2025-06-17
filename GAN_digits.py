# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 08:00:14 2025

@author: IVAN
"""

"""
Ejemplo: GAN sencilla que genere dígitos escritos a mano (como los del dataset MNIST).
Resultado: 
    Al principio, el generador creará imágenes aleatorias (ruido). 
    Con el tiempo (época 0, 50, 100), aprenderá a generar dígitos cada vez más realistas.
"""

# Paso 1: Importar Librerías
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Paso 2: Crear el Generador
def build_generator():
    model = Sequential([
        Dense(128, input_dim=150, activation='relu'), # Recibe ruido aleatorio (100 números)
        Dense(784, activation='sigmoid'),             # Genera una imagen de 28x28 (784 pixels)
                                                      # Se usa Sigmoide ya que los píxeles son normalizados [0, 1]
        Reshape((28, 28))                             # Le da forma de imagen
    ])
    return model

# Paso 3: Crear el Discriminador
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Aplana la imagen
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')   # Decide si es real (1) o falso (0)
    ])
    return model

# Paso 4: Entrenar la GAN
# Cargar datos (MNIST)
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0  # Normalizar imágenes entre 0 y 1

# Crear modelos
generator = build_generator()
discriminator = build_discriminator()

# Compilar el discriminador
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN (generador + discriminador)
discriminator.trainable = False  # Congelar el discriminador al entrenar el generador
gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Entrenamiento
epochs = 200
batch_size = 64

for epoch in range(epochs):
    # Entrenar discriminador
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 150))
    fake_images = generator.predict(noise)
    
    X = np.concatenate([real_images, fake_images])
    y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])  # 1=real, 0=falso
    
    d_loss = discriminator.train_on_batch(X, y)
    
    # Entrenar generador
    noise = np.random.normal(0, 1, (batch_size, 150))
    y_gan = np.ones(batch_size)  # Engañar al discriminador (etiquetas como "reales")
    g_loss = gan.train_on_batch(noise, y_gan)
    
    # Mostrar progreso (cada 10 imágenes)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Disc Loss: {d_loss}, Gen Loss: {g_loss}")
        # Generar ejemplo
        plt.imshow(fake_images[0], cmap='gray')
        plt.show()