# precompute_fid_stats.py
import numpy as np, tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from utils import scale_and_convert_to_rgb
import config, os

(X_train, _), _ = tf.keras.datasets.mnist.load_data()
X = scale_and_convert_to_rgb((X_train.astype(np.float32)-127.5)/127.5,
                             config.INCEPTION_INPUT_SHAPE)

inception = InceptionV3(include_top=False, pooling='avg',
                        input_shape=config.INCEPTION_INPUT_SHAPE,
                        weights='imagenet')
acts = inception.predict(X, batch_size=256, verbose=1)
mu, sigma = acts.mean(axis=0), np.cov(acts, rowvar=False)
np.savez(os.path.join(config.CACHE_DIR, "fid_mnist.npz"),
         mu=mu, sigma=sigma)
