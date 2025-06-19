# -*- coding: utf-8 -*-
"""
Mini-SAGAN para MNIST
=====================
▪ Auto-atención 2-D (SAGAN)
▪ Normalización espectral en el discriminator
▪ LeakyReLU + BatchNorm (buenas prácticas DCGAN)
TensorFlow 2.x
@author: IVAN
"""

import tensorflow as tf
from tensorflow.keras.layers import (Layer, Conv2D, Conv2DTranspose, Dense,
                                     Reshape, Flatten, LeakyReLU,
                                     BatchNormalization, Input,
                                     SpectralNormalization)
from tensorflow.keras.models import Sequential


# ---------------------------------------------------------------------------
# 1.  Capa de Auto-Atención (versión 2-D estilo SAGAN)
# ---------------------------------------------------------------------------
class SelfAttention2D(Layer):
    """
    Auto-atención de imagen (SAGAN).  Fusión canal-espacio:
        • θ, φ  : claves y consultas   (C//8 canales)
        • g     : valores            (C//2 canales)
        • γ     : peso entrenable que empieza en 0
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Proyecciones 1×1
        self.theta = Conv2D(channels // 8, kernel_size=1, padding='same')
        self.phi   = Conv2D(channels // 8, kernel_size=1, padding='same')
        self.g     = Conv2D(channels // 2, kernel_size=1, padding='same')
        self.o     = Conv2D(channels,      kernel_size=1, padding='same')
        # Escalar entrenable (γ) para balancear la rama de atención
        self.gamma = self.add_weight(name='gamma', shape=[1],
                                     initializer='zeros', trainable=True)

    def call(self, x):
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], self.channels
        θ = tf.reshape(self.theta(x), [b, -1, c // 8])      # (B, HW, C/8)
        φ = tf.reshape(self.phi(x),   [b, -1, c // 8])      # (B, HW, C/8)
        g = tf.reshape(self.g(x),     [b, -1, c // 2])      # (B, HW, C/2)

        β = tf.nn.softmax(tf.matmul(θ, φ, transpose_b=True))  # atención (HW × HW)
        o = tf.matmul(β, g)                                   # (B, HW, C/2)
        o = tf.reshape(o, [b, h, w, c // 2])
        o = self.o(o)
        return self.gamma * o + x                             # Residual


# ---------------------------------------------------------------------------
# 2.  Generador con auto-atención (resolución 28×28)
# ---------------------------------------------------------------------------
def build_sagan_generator(latent_dim=128):
    model = Sequential(name="sagan_generator")

    # Latent → 7×7×256
    model.add(Input(shape=(latent_dim,)))
    model.add(Dense(7 * 7 * 256, use_bias=False))
    model.add(Reshape((7, 7, 256)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # Auto-atención a baja resolución
    model.add(SelfAttention2D(256))

    # Upsample 14×14
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same',
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # Auto-atención intermedia
    model.add(SelfAttention2D(128))

    # Upsample 28×28
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same',
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # Salida 28×28×1 (tanh para mapear a [-1,1])
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same',
                              activation='tanh'))
    return model


# ---------------------------------------------------------------------------
# 3.  Discriminador con Normalización Espectral + Auto-Atención
# ---------------------------------------------------------------------------
SpectralConv2D = lambda *a, **k: SpectralNormalization(
    Conv2D(*a, **k), power_iterations=1)

SpectralDense  = lambda units: SpectralNormalization(Dense(units))

def build_sagan_discriminator(img_shape=(28, 28, 1)):
    model = Sequential(name="sagan_discriminator")

    model.add(Input(shape=img_shape))

    # 28×28 → 14×14
    model.add(SpectralConv2D(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))

    # Auto-atención temprana
    model.add(SelfAttention2D(64))

    # 14×14 → 7×7
    model.add(SpectralConv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))

    # 7×7 → 4×4
    model.add(SpectralConv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(SpectralDense(1))  # logits (sin activación sigmoide)

    return model