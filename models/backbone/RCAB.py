import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, GlobalAveragePooling2D, add, multiply


def rcab(x):
    identity = x
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same', )(x)
    x = ReLU()(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same', )(x)
    # channel attention
    ca = GlobalAveragePooling2D(keepdims=True)(x)
    ca = Conv2D(filters=4,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same', )(ca)
    ca = ReLU()(ca)
    ca = Conv2D(filters=64,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                activation='sigmoid',)(ca)
    x = multiply([x, ca])
    x = add([x, identity])
    return x
