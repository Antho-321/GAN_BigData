import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os
## NUEVO ##
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Configuración
np.random.seed(42)
tf.random.set_seed(42)

latent_dim = 100
img_shape = (28, 28, 1)
epochs = 2000
batch_size = 64
save_interval = 200

# Crear carpeta de imágenes generadas
os.makedirs("generated_images", exist_ok=True)

# ======================
# Carga de datos
# ======================
(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

# ======================
# Definir modelos (Generador y Discriminador)
# (Se mantiene el código original para build_generator, build_discriminator y build_gan)
# ======================

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(7*7*128, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(64, 5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(32, 5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, 5, strides=2, padding='same', use_bias=False, activation='tanh'))
    return model

def build_discriminator():
    model = models.Sequential()
    model.add(layers.Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    z = layers.Input(shape=(latent_dim,))
    img = generator(z)
    validity = discriminator(img)
    return models.Model(z, validity)

# ======================
# Crear y compilar modelos
# ======================

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')


## NUEVO ##
# ======================
# Función FID y preparación
# ======================

# Cargar el modelo InceptionV3 pre-entrenado en ImageNet
# Se usa un input_shape de 75x75x3, ya que es el mínimo aceptado
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))

def scale_and_convert_to_rgb(images):
    """ Redimensiona imágenes a 75x75 y las convierte a 3 canales (RGB) """
    # Reescalar las imágenes de [-1, 1] a [0.0, 255.0] y mantener el tipo float
    images_rescaled = (images + 1) * 127.5  # This keeps it as a float32 tensor
    
    # Redimensionar a 75x75
    images_resized = tf.image.resize(images_rescaled, (75, 75), method='nearest')
    
    # Convertir de escala de grises a RGB
    images_rgb = tf.image.grayscale_to_rgb(images_resized)
    
    return images_rgb

def calculate_fid(model, images1, images2):
    """
    Calcula el Fréchet Inception Distance (FID) entre dos grupos de imágenes.
    """
    # Preprocesar imágenes para InceptionV3
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

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

# ======================
# Funciones auxiliares y entrenamiento
# ======================

def sample_images(epoch, grid=4):
    noise = np.random.normal(0, 1, (grid * grid, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5 # Reescalar a [0, 1]

    fig, axs = plt.subplots(grid, grid, figsize=(5, 5))
    count = 0
    for i in range(grid):
        for j in range(grid):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    plt.suptitle(f"Imágenes generadas - Época {epoch}")
    fig.savefig(f"generated_images/epoch_{epoch}.png")
    # plt.show() # Se puede comentar para que no interrumpa el entrenamiento
    plt.close()

def train(epochs, batch_size, sample_interval):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs + 1):
        # Entrenar Discriminador
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Entrenar Generador
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # Si es el intervalo para guardar y evaluar
        if epoch % sample_interval == 0:
            print(f"[{epoch}] D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}% | G loss: {g_loss:.4f}", end="")
            sample_images(epoch)

            ## NUEVO: Calcular FID ##
            # 1. Preparar imágenes reales y generadas (usando más imágenes para un cálculo más estable)
            fid_batch_size = 512 # Usar un batch más grande para FID
            idx_fid = np.random.randint(0, X_train.shape[0], fid_batch_size)
            real_fid_imgs = X_train[idx_fid]

            noise_fid = np.random.normal(0, 1, (fid_batch_size, latent_dim))
            gen_fid_imgs = generator.predict(noise_fid, verbose=0)

            # 2. Convertir imágenes al formato de InceptionV3
            real_fid_imgs_rgb = scale_and_convert_to_rgb(real_fid_imgs)
            gen_fid_imgs_rgb = scale_and_convert_to_rgb(gen_fid_imgs)

            # 3. Calcular FID
            fid_score = calculate_fid(inception_model, real_fid_imgs_rgb, gen_fid_imgs_rgb)
            print(f" | FID: {fid_score:.2f}")


# Iniciar entrenamiento
train(epochs=epochs, batch_size=batch_size, sample_interval=save_interval)