# model.py

from tensorflow.keras import layers, models

def build_generator(latent_dim):
    """Construye el modelo Generador para la DCGAN."""
    model = models.Sequential(name="Generator")
    model.add(layers.Dense(7 * 7 * 128, use_bias=False, input_shape=(latent_dim,)))
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

def build_discriminator(img_shape):
    """Construye el modelo Discriminador para la DCGAN."""
    model = models.Sequential(name="Discriminator")
    model.add(layers.Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model