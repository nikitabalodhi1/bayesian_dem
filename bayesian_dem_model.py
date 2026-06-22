"""
bayesian_dem_model.py
================
Bayesian Convolutional Neural Network for Solar Differential Emission Measure (DEM) Inference.

Overview
--------
This script builds an approximate Bayesian CNN that infers
the Differential Emission Measure (DEM) from multi-channel AIA (Atmospheric Imaging
Assembly) solar EUV images.

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# %%
# =============================================================================
# Constants and Configuration
# =============================================================================

# Logarithmic temperature grid: log10(T/K) from 5.5 to 7.2 in 18 steps
# Each element represents the centre of one temperature bin used for DEM reconstruction
LOGT = np.linspace(5.5, 7.2, 18)

# Width of each temperature bin in log10 space
TRESP_DLOGT = np.full(18, 0.1)

# Conversion factor: Due to the caluclation of DEM in logT scale
FACTOR = 10**LOGT * np.log(10**TRESP_DLOGT)

# ── AIA temperature-response kernel ──────────────────────────────────────────
# k_matrix shape: (6, 18) — six AIA EUV channels × 18 temperature bins.

k_matrix = np.load("/Users/nikitabalodhi/Documents/data_prepped/k_matrix.npy")
kernel = np.expand_dims(np.expand_dims(np.transpose(k_matrix), axis=0), axis=0)


# %%
# =============================================================================
# Scaling Utilities
# =============================================================================

def em_scale(x):
    """
    Scale emission measure (EM) values for neural network input.
 
    Applies a square-root compression after clipping negative values to zero,
    normalised by 1e19 to bring values into a numerically convenient range.
    """
    x = np.where(x <= 0.0, 0.0, x)
    return np.sqrt(x / 1e19)
 
 
def em_unscale(x):
    """
    Invert :func:`em_scale` to recover physical emission measure values.
    """
    return 1e19 * (x * x)
 
 
def img_scale(x):
    """
    Scale AIA intensity values (DN/s) for neural network input.
 
    Applies square-root compression after clipping negative values to zero.
    """
    x = np.where(x <= 0.0, 0.0, x)
    return np.sqrt(x)
 
 
def img_unscale(x):
    """
    Invert :func:`img_scale` to recover DN/s values.
    """
    return x * x


# %%
# =============================================================================
# Custom Keras Layers
# =============================================================================
 
class MonteCarloDropout(layers.Dropout):
    """
    Dropout layer that remains active at inference time.
 
    Standard Keras Dropout is disabled during prediction. This subclass
    overrides that behaviour so that stochastic forward passes can be used
    for Bayesian uncertainty estimation (Gal & Ghahramani, 2016).
    """
 
    def call(self, inputs):
        # Force training=True so dropout is never bypassed
        return super().call(inputs, training=True)
 
 
class LearnableNoise(tf.keras.layers.Layer):
    """
    Spatially-resolved learnable noise map.
 
    Produces a per-pixel, per-channel standard deviation estimate for an assumed gaussian noise that is trained
    jointly with the rest of the network. A softplus activation ensures
    positivity.
 
    Parameters
    ----------
    shape : tuple
        Shape of the std map (height, width, channels), matching the
        spatial dimensions of the AIA input tiles.
    """
 
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
 
    def build(self, input_shape):
        self.noise_map = self.add_weight(
            shape=self.shape,
            initializer="zeros",
            trainable=True,
            name="learnable_noise_map",
        )
 
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Expand and tile across the batch dimension
        return tf.tile(
            tf.expand_dims(tf.nn.softplus(self.noise_map), axis=0),
            [batch_size, 1, 1, 1],
        )
 
 
# =============================================================================
# Loss Function
# =============================================================================
 
def gaussian_nll_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    noise_pred: tf.Tensor,
    eps: float = 1e-5,
) -> tf.Tensor:
    """
    Gaussian negative log-likelihood loss with learned standard deviation.
 
    Encourages the network to predict AIA noise by
    penalising both prediction error and standard deviation estimates
 
    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth AIA images.
    y_pred : tf.Tensor
        Predicted AIA images.
    noise_pred : tf.Tensor
        Predicted (positive) standard deviation map.
    eps : float
        Small constant for numerical stability
 
    Returns
    -------
    tf.Tensor
        Scalar mean NLL loss.
    """
    y_true    = tf.cast(y_true,    tf.float32)
    y_pred    = tf.cast(y_pred,    tf.float32)
    noise_pred  = tf.cast(noise_pred,  tf.float32)
 
    sq_error = tf.square(y_true - y_pred)
    nll = (sq_error / (2.0 * tf.square(noise_pred)+eps)) + 0.5 * tf.math.log(tf.square(noise_pred)+eps)
    return tf.reduce_mean(nll)

# =============================================================================
# Model Architecture
# =============================================================================

def build_model():
    """
    Build the Bayesian CNN for DEM inference.
 
    Architecture overview
    ---------------------
    - A 1×1 convolution maps 6 AIA channels to a 128-feature representation.
    - Monte Carlo Dropout (p=0.3) is applied after each major layer.
    - A transposed convolution upsamples spatial features.
    - A 3×3 convolution produces the 18-temperature DEM map.
    - A binary mask (supplied at training time) zeros out invalid pixels.
    - The response layer synthesises AIA images from the predicted DEM for
      the noise aware reconstruction loss.
    - A learnable std map provides per-pixel uncertainty estimates for an assumed gaussian noise on the AIA images.
 
    Two model handles are returned:
    - ``train_model``: accepts ``[image, mask]`` inputs and outputs
      ``{output_dem, output_img}``; includes the std regularisation
      as an added loss.
    - ``test_model``: accepts only the image input and additionally outputs
      the std map ``output_std``.
 
    Parameters
    ----------
    kernel : np.ndarray
        AIA temperature-response kernel, shape (1, 1, 18, 6).
    factor : np.ndarray
        Per-temperature EM integration factor, shape (18,).
 
    Returns
    -------
    train_model : tf.keras.Model
    test_model  : tf.keras.Model
    """

    # Shared regulariser kwargs for conv layers
    conv1x1_kwargs = {"kernel_size": 1, "kernel_regularizer": l2(1e-6)}
    conv3x3_kwargs = {"kernel_size": 3, "kernel_regularizer": l2(1e-6)}

    # ── Inputs ────────────────────────────────────────────────────────────────
    image_input = layers.Input((512, 512, 6),  name="image_input")
    mask_input  = layers.Input((512, 512, 18), name="mask_input")
    # ──────────────────────────────────────────────────────────────────────────

    # 1×1 conv: mixes the six AIA channels into a 128-feature representation.
    x = layers.Conv2D(128, **conv1x1_kwargs,
                      activation=keras.layers.LeakyReLU(alpha=0.1))(image_input)
    x = MonteCarloDropout(0.3)(x)

    # Transposed conv: expands spatial context slightly while reducing channels.
    x = layers.Conv2DTranspose(64, kernel_size=3,
                               activation=keras.layers.LeakyReLU(alpha=0.1))(x)
    x = MonteCarloDropout(0.3)(x)

    # ── DEM output ────────────────────────────────────────────────────────────
    # Produce 18-channel DEM map (one channel per temperature bin).
    dem_unmasked = layers.Conv2D(
        18, **conv3x3_kwargs,
        activation=keras.layers.LeakyReLU(alpha=0.1),
        name="output_dem_unmasked",
    )(x)

    # Zero out pixels with no valid reference DEM (used during training only).
    dem_masked = layers.multiply([dem_unmasked, mask_input], name="output_dem")
    # ──────────────────────────────────────────────────────────────────────────

    # ── Physics-constrained image reconstruction ──────────────────────────────
    # Step 1: convert scaled DEM → physical emission measure [cm^-5]
    #         by inverting em_scale and multiplying by the temperature bin factor.
    unscale = layers.Lambda(lambda x: em_unscale(x) * FACTOR)(dem_unmasked)

    # Step 2: convolve the EM map with the AIA temperature-response kernel to
    #         produce synthetic AIA intensities (DN/s). Kernel is fixed (non-trainable).
    output_img_unscaled = layers.Lambda(
        lambda x: tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    )(unscale)

    # Step 3: apply sqrt scaling to match the img_scale convention used on inputs.
    output_img = layers.Lambda(lambda x: tf.sqrt(x), name="output_img")(output_img_unscaled)
    # ──────────────────────────────────────────────────────────────────────────

    # ── Learnable aleatoric noise map ─────────────────────────────────────────
    # Per-pixel per-channel std estimate for the Gaussian NLL image loss.
    output_std = LearnableNoise(shape=(512, 512, 6), name="output_std")(dem_unmasked)

    # ── Assemble models ───────────────────────────────────────────────────────
    # Training model: takes masked DEM and reconstructed image as outputs.
    # The mean of output_std is added as an auxiliary L1 regulariser that
    # discourages the network from predicting arbitrarily large noise values.
    train_model = Model(
        inputs=[image_input, mask_input],
        outputs={"output_dem": dem_masked, "output_img": output_img},
    )
    train_model.add_loss(tf.reduce_mean(output_std))

    # Test / inference model: uses unmasked DEM so all pixels are available,
    # and additionally exposes the aleatoric uncertainty map.
    test_model = Model(
        inputs=image_input,
        outputs={
            "output_dem": dem_unmasked,
            "output_img": output_img,
            "output_std": output_std,
        },
    )

    return train_model, test_model