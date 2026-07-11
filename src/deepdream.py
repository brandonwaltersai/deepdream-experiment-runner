"""DeepDream engine on a pretrained InceptionV3 (ImageNet). TensorFlow implementation.

Two variants:
  - Simple: single-scale gradient ascent. Fast, good for small images.
  - Multi-octave + tiled: processes the image at multiple scales and in
    tiles, which is what keeps memory bounded on large images and produces
    more coherent patterns across scales (the tutorial-standard approach).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def load_base_model() -> tf.keras.Model:
    model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    model.trainable = False
    return model


def available_layer_names(base_model: tf.keras.Model) -> set[str]:
    return {l.name for l in base_model.layers}


def build_dream_model(base_model: tf.keras.Model, layer_names: list[str]) -> tf.keras.Model:
    available = available_layer_names(base_model)
    missing = [n for n in layer_names if n not in available]
    if missing:
        mixed_layers = [n for n in available if n.startswith("mixed")]
        raise ValueError(f"Invalid layer name(s): {missing}. Example valid names: {sorted(mixed_layers)[:5]}")
    outputs = [base_model.get_layer(n).output for n in layer_names]
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)


def deprocess(img_f32: tf.Tensor) -> tf.Tensor:
    """[-1, 1] float -> uint8."""
    img = 255 * (img_f32 + 1.0) / 2.0
    return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)


def preprocess_uint8(img_uint8: np.ndarray) -> tf.Tensor:
    img = tf.keras.applications.inception_v3.preprocess_input(img_uint8)
    return tf.convert_to_tensor(img)


def calc_loss(img: tf.Tensor, dream_model: tf.keras.Model) -> tf.Tensor:
    img_batch = tf.expand_dims(img, axis=0)
    acts = dream_model(img_batch)
    if not isinstance(acts, (list, tuple)):
        acts = [acts]
    return tf.add_n([tf.reduce_mean(a) for a in acts])


class DeepDream(tf.Module):
    """Single-scale gradient ascent on the chosen layer activations."""

    def __init__(self, dream_model: tf.keras.Model):
        super().__init__()
        self.dream_model = dream_model

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
    ))
    def __call__(self, img, steps, step_size):
        loss = tf.constant(0.0)
        for _ in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.dream_model)
            grads = tape.gradient(loss, img)
            grads /= tf.math.reduce_std(grads) + 1e-8
            img = tf.clip_by_value(img + grads * step_size, -1, 1)
        return loss, img


def run_deep_dream_simple(img_uint8: np.ndarray, dream_model: tf.keras.Model,
                           steps: int = 100, step_size: float = 0.01) -> tuple[np.ndarray, float]:
    img = preprocess_uint8(img_uint8)
    dd = DeepDream(dream_model)
    loss, out = dd(img, tf.constant(int(steps)), tf.constant(float(step_size)))
    return deprocess(out).numpy(), float(loss.numpy())


def _random_roll(img, maxroll):
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    return shift, tf.roll(img, shift=shift, axis=[0, 1])


class TiledGradients(tf.Module):
    """Computes DeepDream gradients tile-by-tile so large images stay within memory."""

    def __init__(self, dream_model: tf.keras.Model):
        super().__init__()
        self.dream_model = dream_model

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[2], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ))
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = _random_roll(img, tile_size)  # jitter reduces tile-seam artifacts
        gradients = tf.zeros_like(img_rolled)
        h, w = img_size[0], img_size[1]

        for y in tf.range(0, h, tile_size):
            for x in tf.range(0, w, tile_size):
                y1, x1 = tf.minimum(y + tile_size, h), tf.minimum(x + tile_size, w)
                with tf.GradientTape() as tape:
                    tape.watch(img_rolled)
                    tile = img_rolled[y:y1, x:x1]
                    loss = calc_loss(tile, self.dream_model)
                gradients += tape.gradient(loss, img_rolled)

        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        return gradients


def run_deep_dream_with_octaves(img_uint8: np.ndarray, dream_model: tf.keras.Model,
                                 steps_per_octave: int = 100, step_size: float = 0.01,
                                 octaves=range(-2, 3), octave_scale: float = 1.3,
                                 tile_size: int = 512) -> np.ndarray:
    img = preprocess_uint8(img_uint8)
    base_shape = tf.shape(img)[:-1]
    float_base = tf.cast(base_shape, tf.float32)
    tiled = TiledGradients(dream_model)

    for octave in octaves:
        new_size = tf.cast(float_base * (octave_scale ** octave), tf.int32)
        img = tf.image.resize(img, new_size)
        for _ in tf.range(int(steps_per_octave)):
            grads = tiled(img, new_size, tf.constant(int(tile_size)))
            img = tf.clip_by_value(img + grads * float(step_size), -1, 1)

    img = tf.image.resize(img, base_shape)
    return deprocess(img).numpy()
