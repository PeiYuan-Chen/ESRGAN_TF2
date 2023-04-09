import tensorflow as tf
from tensorflow.keras.layers import add
from models.backbone.RDB import residual_dense_block


def residual_in_residual_dense_block(x, kernel_initializer=tf.keras.initializers.GlorotNormal()):
    identity = x
    for _ in range(3):
        x = residual_dense_block(x, kernel_initializer=kernel_initializer)
    x = x * 0.2
    return add([x, identity])
