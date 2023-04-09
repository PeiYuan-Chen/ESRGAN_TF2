import tensorflow as tf


def rgb_to_ycbcr_y(image):
    image = tf.cast(image, tf.float32) / 255.0
    y = tf.tensordot(image, [65.481, 128.553, 24.966], axes=[-1, -1]) + 16.0
    y = tf.round(y)
    y = tf.expand_dims(y, axis=-1)
    return y
