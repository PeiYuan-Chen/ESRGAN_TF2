import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, ReLU, add


def residual_block_crc(x, filter_num=64, kernel_initializer=tf.keras.initializers.glorot_normal()):
    inputs = x
    x = Conv2D(filters=filter_num,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = ReLU()(x)
    x = Conv2D(filters=filter_num,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               kernel_initializer=kernel_initializer)(x)
    if inputs.shape[-1] == filter_num:
        short_cut = inputs
    else:
        short_cut = Conv2D(filters=filter_num,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='same',
                           use_bias=False,
                           kernel_initializer=kernel_initializer)(inputs)
    return add([x, short_cut])
