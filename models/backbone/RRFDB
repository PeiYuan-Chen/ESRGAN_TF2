import tensorflow as tf
from tensorflow.keras.layers import add
from models.backbone.RFDB import receptive_field_dense_block


def residual_of_receptive_field_dense_block(x, channels: int = 64, growth_channels: int = 32,
                                            residual_scaling: float = 0.2,
                                            kernel_initializer=tf.keras.initializers.GlorotNormal()):
    identity = x
    for _ in range(3):
        x = receptive_field_dense_block(
            x, channels, growth_channels, residual_scaling, kernel_initializer)
    x = x * residual_scaling
    return add([x, identity])
