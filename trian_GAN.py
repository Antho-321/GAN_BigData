import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os

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
# Visualizar imágenes reales
# ======================
(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

num_images = 16
indices = np.random.randint(0, X_train.shape[0], num_images)
sample_images_real = X_train[indices]
sample_labels = y_train[indices]

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images_real[i, :, :, 0], cmap='gray')
    ax.set_title(f"{sample_labels[i]}", fontsize=8)
    ax.axis('off')
plt.suptitle("Ejemplos reales del dataset MNIST")
plt.tight_layout()
plt.show()

# ======================
# Definir modelos
# ======================

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(7*7*128, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 128)))  # (7, 7, 128)
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

# ======================
# Entrenamiento
# ======================

def sample_images(epoch, grid=4):
    noise = np.random.normal(0, 1, (grid * grid, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Reescalar a [0, 1]

    fig, axs = plt.subplots(grid, grid, figsize=(5, 5))
    count = 0
    for i in range(grid):
        for j in range(grid):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    plt.suptitle(f"Imágenes generadas - Época {epoch}")
    fig.savefig(f"generated_images/epoch_{epoch}.png")
    plt.show()
    plt.close()

def train(epochs, batch_size, sample_interval):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs + 1):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        if epoch % sample_interval == 0:
            print(f"[{epoch}] D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}% | G loss: {g_loss:.4f}")
            sample_images(epoch)

# Iniciar entrenamiento
train(epochs=epochs, batch_size=batch_size, sample_interval=save_interval)
