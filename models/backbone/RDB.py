import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, concatenate, add


def residual_dense_block(x, kernel_initializer=tf.keras.initializers.GlorotNormal()):
    identity = x
    out1 = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(x)
    out1 = LeakyReLU(alpha=0.2)(out1)
    out2 = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(concatenate([x, out1]))
    out2 = LeakyReLU(alpha=0.2)(out2)
    out3 = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(concatenate([x, out1, out2]))
    out3 = LeakyReLU(alpha=0.2)(out3)
    out4 = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(concatenate([x, out1, out2, out3]))
    out4 = LeakyReLU(alpha=0.2)(out4)
    out5 = Conv2D(filters=64,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(concatenate([x, out1, out2, out3, out4]))
    out = out5 * 0.2
    return add([identity, out])


def dense_block(x, kernel_initializer=tf.keras.initializers.GlorotNormal(), residual_scaling=0.2):
    out1 = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(x)
    out1 = LeakyReLU(alpha=0.2)(out1)
    out2 = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(concatenate([x, out1]))
    out2 = LeakyReLU(alpha=0.2)(out2)
    out3 = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(concatenate([x, out1, out2]))
    out3 = LeakyReLU(alpha=0.2)(out3)
    out4 = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(concatenate([x, out1, out2, out3]))
    out4 = LeakyReLU(alpha=0.2)(out4)
    out5 = Conv2D(filters=64,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same',
                  kernel_initializer=kernel_initializer, )(concatenate([x, out1, out2, out3, out4]))
    out = out5 * residual_scaling
    return out
