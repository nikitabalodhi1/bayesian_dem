# %%

"""

This script trains, and validates the approximate Bayesian model defined in 
bayesian_model.py and saves the best weights

The pipeline:
    1. Loads AIA intensity images and reference DEM maps for three observation dates.
    2. Scales inputs and targets for efficient training.
    3. Constructs a Bayesian CNN with Monte Carlo Dropout for epistemic uncertainty
       and a learnable spatially-resolved noise map for aleatoric uncertainty.
    4. Trains with a combined loss:
         - MAE on the DEM prediction.
         - Gaussian NLL on the noise aware AIA image reconstruction.
    5. Applies early stopping and saves the best model weights.

"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from bayesian_dem_model import build_model, img_scale, em_scale, img_unscale, em_unscale, gaussian_nll_loss

# =============================================================================
# Data Loading and Preparation
# =============================================================================
# Three observation dates are used:
#   - 2010-09-17: quiet-Sun / near-minimum conditions
#   - 2014-04-20: active-region
#   - 2014-05-18: active-region
#
# Each .npy file contains 64 cutouts of shape (512, 512, 6) from full res (lev1.5) preprocessed AIA images
# and (512, 512, 18) for DEM maps
# 144 image pairs are used for training; the remaining 48 for validation.

aia_1 = np.load('/2010-09-17_split.npy')
aia_2 = np.load('/2014-04-20_split.npy')
aia_3 = np.load('/2014-05-18_split.npy')
dem_1 = np.load('/2010-09-17_split_dem.npy')
dem_2 = np.load('/2014-04-20_split_dem.npy')
dem_3 = np.load('/2014-05-18_split_dem.npy')

# Concatenate across dates and apply scaling transforms.
X_train = img_scale(np.concatenate([aia_1[0:48], aia_2[0:48], aia_3[0:48]], 0))  # (144, 512, 512, 6)
y_train = em_scale(np.concatenate([dem_1[0:48], dem_2[0:48], dem_3[0:48]], 0))   # (144, 512, 512, 18)
X_val   = img_scale(np.concatenate([aia_1[48:],  aia_2[48:],  aia_3[48:]],  0))  # ( 48, 512, 512, 6)
y_val   = em_scale(np.concatenate([dem_1[48:],  dem_2[48:],  dem_3[48:]],  0))   # ( 48, 512, 512, 18)

# Build binary masks: 1 where the reference DEM is non-zero (valid pixel),
# 0 elsewhere (e.g. off-disk regions where no DEM inversion was performed).
mask = np.ones(y_train.shape)
mask[np.where(y_train == 0)] = 0

mask_val = np.ones(y_val.shape)
mask_val[np.where(y_val == 0)] = 0


# %%
# =============================================================================
# tf.data Pipeline
# =============================================================================
# Each dataset item is a tuple:
#   inputs  = (AIA image tile, binary mask)
#   targets = (DEM map, AIA image tile)          ← image used for NLL loss

BATCH_SIZE = 1

train_dataset = tf.data.Dataset.from_tensor_slices(((X_train, mask),     (y_train, X_train)))
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset   = tf.data.Dataset.from_tensor_slices(((X_val, mask_val), (y_val, X_val)))
val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# %%

#Define model
train_model,_ = build_model()


# =============================================================================
# Training and Validation Steps
# =============================================================================
# A sub-model is constructed to expose the LearnableNoise output from the
# training model, since it is used directly in the Gaussian NLL loss but is
# not part of train_model's declared outputs.

output_std_layer = train_model.get_layer('output_std')
train_model_output_std = tf.keras.Model(
    inputs=train_model.inputs,
    outputs=output_std_layer.output,
)


@tf.function
def train_step(model, x_batch, y_batch):
    """
    Execute one gradient-update step on a single batch

    Computes the combined loss:
        total_loss = 0.3 * dem_loss + 0.7 * img_loss

    where:
        - dem_loss : mean absolute error between predicted and reference DEM
        - img_loss : Gaussian NLL between the AIA images
          and the ground-truth AIA images, using the learnable noise map

    The weighted loss can be changed by adjusting the coefficients (0.3 and 0.7)

    Parameters
    ----------
    model : tf.keras.Model
        The training model (train_model).
    x_batch : tuple of tf.Tensor
        ``(image_tile, mask_tile)`` — scaled AIA image and validity mask.
    y_batch : tuple of tf.Tensor
        ``(dem_tile, image_tile)`` — scaled reference DEM and scaled AIA image.

    Returns
    -------
    dem_loss : tf.Tensor
        Scalar DEM MAE for this batch.
    img_loss : tf.Tensor
        Scalar image Gaussian NLL for this batch.
    """
    x_input, mask_input = x_batch
    y_dem_true, y_img_true = y_batch

    with tf.GradientTape() as tape:
        y_pred = model([x_input, mask_input], training=True)

        # MAE between predicted (masked) DEM and reference DEM.
        dem_loss = tf.reduce_mean(
            tf.keras.losses.mean_absolute_error(y_dem_true, y_pred['output_dem'])
        )

        # Retrieve aleatoric std map for the NLL loss.
        output_std = train_model_output_std([x_input, mask_input], training=True)

        # Gaussian NLL between physics-reconstructed images and ground-truth AIA.
        img_loss = gaussian_nll_loss(y_img_true, y_pred['output_img'], output_std)

        total_loss = 0.3 * dem_loss + 0.7 * img_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return dem_loss, img_loss


@tf.function
def val_step(model, x_batch, y_batch):
    """
    Evaluate the model on a single validation batch (no gradient update).

    Parameters
    ----------
    model : tf.keras.Model
        The training model (train_model).
    x_batch : tuple of tf.Tensor
        ``(image_tile, mask_tile)``.
    y_batch : tuple of tf.Tensor
        ``(dem_tile, image_tile)``.

    Returns
    -------
    dem_loss : tf.Tensor
        Scalar DEM MAE for this batch.
    img_loss : tf.Tensor
        Scalar image Gaussian NLL for this batch.
    """
    x_input, mask_input = x_batch
    y_dem_true, y_img_true = y_batch

    # training=False: standard Dropout is off, but MonteCarloDropout stays on.
    y_pred = model([x_input, mask_input], training=False)
    dem_loss = tf.reduce_mean(
        tf.keras.losses.mean_absolute_error(y_dem_true, y_pred['output_dem'])
    )

    output_std = train_model_output_std([x_input, mask_input], training=False)
    img_loss = gaussian_nll_loss(y_img_true, y_pred['output_img'], output_std)
    return dem_loss, img_loss


# %%
# =============================================================================
# Optimizer Configuration
# =============================================================================

# MultiOptimizer (TFA) allows pairing each optimizer with a subset of layers:
#   - optimizers[0] → layers 0–6  (input, DEM map)
#   - optimizers[1] → layers 7+   (DEM head, reconstruction, noise map)
# The learning rate and weight decay for each optimizer can be tuned independently
optimizers = [
    tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-9),
    tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-9),
]

optimizers_and_layers = [
    (optimizers[0], train_model.layers[0:7]),
    (optimizers[1], train_model.layers[7:]),
]

optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)


def compile_model(model, optimizer):
    """
    Compile the model with the given optimizer

    Parameters
    ----------
    model : tf.keras.Model
    optimizer : tf.keras.optimizers.Optimizer

    Returns
    -------
    tf.keras.Model

    """
    model.compile(optimizer=optimizer)
    return model


compile_model(train_model, optimizer)


# %%
# =============================================================================
# Training Loop with Early Stopping
# =============================================================================

# Best weights are saved to disk whenever val_loss improves.
# Training halts if no improvement is observed for PATIENCE consecutive epochs.

EPOCHS   = 250   # Maximum number of training epochs.
PATIENCE = 60    # Early-stopping patience (epochs without improvement).

best_val_loss = float("inf")
wait = 0

with tf.device("/device:GPU:0"):
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Accumulate per-epoch metrics.
        train_dem = tf.keras.metrics.Mean()
        train_img = tf.keras.metrics.Mean()
        val_dem   = tf.keras.metrics.Mean()
        val_img   = tf.keras.metrics.Mean()

        # ── Training pass ──────────────────────────────────────────────────
        for xb, yb in train_dataset:
            d, i = train_step(train_model, xb, yb)
            train_dem.update_state(d)
            train_img.update_state(i)

        # ── Validation pass ────────────────────────────────────────────────
        for xb, yb in val_dataset:
            d, i = val_step(train_model, xb, yb)
            val_dem.update_state(d)
            val_img.update_state(i)

        print(f"Train DEM: {train_dem.result():.4f}, IMG: {train_img.result():.4f}")
        print(f"Val   DEM: {val_dem.result():.4f}, IMG: {val_img.result():.4f}")

        # ── Checkpoint and early stopping ──────────────────────────────────
        current_loss = 0.3 * val_dem.result() + 0.7 * val_img.result()

        if current_loss <= best_val_loss:
            best_val_loss = current_loss
            wait = 0
            # Save best weights; these can be reloaded into train_model or test_model
            train_model.save_weights(
                "/dropout_weights.h5"
            )
        else:
            wait += 1
            print(f"No improvement for {wait} epoch(s)")

        if wait >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break