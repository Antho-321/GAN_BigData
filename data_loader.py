# data_loader.py

from tensorflow.keras.datasets import mnist
import numpy as np # <-- Añadir import

# CAMBIAR la función load_and_preprocess_mnist
def load_and_preprocess_mnist(normalization_range='0_to_1'):
    """
    Carga el dataset MNIST y lo normaliza.

    Args:
        normalization_range (str): '0_to_1' para normalizar a [0, 1]
                                     '-1_to_1' para normalizar a [-1, 1].
    
    Returns:
        numpy.ndarray: El conjunto de entrenamiento de imágenes MNIST.
    """
    (X_train, _), (_, _) = mnist.load_data()
    
    # Convertir a float32 para la normalización
    X_train = X_train.astype('float32')

    if normalization_range == '0_to_1':
        # Normalizar a [0, 1]
        X_train = X_train / 255.0
    elif normalization_range == '-1_to_1':
        # Normalizar a [-1, 1]
        X_train = (X_train - 127.5) / 127.5
    else:
        raise ValueError("El rango de normalización debe ser '0_to_1' o '-1_to_1'")
        
    # (1) Añadir dimensión de canal aquí mismo
    X_train = np.expand_dims(X_train, -1)      # (N,28,28,1)
    
    # (2) Barajar por si luego usas tf.data
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    
    return X_train.astype("float32")