# -*- coding: utf-8 -*-
"""
Model architectures for a simple GAN and an improved DCGAN.
To be used for generating MNIST-like handwritten digits.

@author: IVAN
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dense, Flatten,
                                     LeakyReLU, Reshape)
from tensorflow.keras.models import Sequential

# =============================================================================
# Architecture 1: Simple GAN
# Based on the first code example provided.
# =============================================================================

def build_simple_generator(latent_dim=150):
    """
    Builds a simple Generator model.

    Args:
        latent_dim (int): The dimension of the input noise vector.

    Returns:
        A Keras Sequential model for the generator.
    """
    model = Sequential([
        Dense(128, input_dim=latent_dim, activation='relu'),
        Dense(784, activation='sigmoid'),
        Reshape((28, 28))
    ], name="simple_generator")
    return model

def build_simple_discriminator(img_shape=(28, 28)):
    """
    Builds a simple Discriminator model.

    Args:
        img_shape (tuple): The shape of the input image (e.g., (28, 28)).

    Returns:
        A Keras Sequential model for the discriminator.
    """
    model = Sequential([
        Flatten(input_shape=img_shape),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name="simple_discriminator")
    return model


# =============================================================================
# Architecture 2: Improved GAN
# Based on the second code example with LeakyReLU and Batch Normalization.
# =============================================================================

def build_improved_generator(latent_dim=200, img_shape=(28, 28)):
    """
    Builds an improved Generator model using LeakyReLU and BatchNormalization.

    Args:
        latent_dim (int): The dimension of the input noise vector.
        img_shape (tuple): The shape of the output image.

    Returns:
        A Keras Sequential model for the generator.
    """
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
        Dense(np.prod(img_shape), activation='sigmoid'),
        Reshape(img_shape)
    ], name="improved_generator")
    return model

def build_improved_discriminator(img_shape=(28, 28)):
    """
    Builds an improved Discriminator model using LeakyReLU.

    Args:
        img_shape (tuple): The shape of the input image (e.g., (28, 28)).

    Returns:
        A Keras Sequential model for the discriminator.
    """
    model = Sequential([
        Flatten(input_shape=img_shape),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ], name="improved_discriminator")
    return model