import tensorflow as tf
from tensorflow.keras.layers import add, concatenate, Conv2D
from models.backbone.RDB import dense_block


def rfa_3(x, kernel_initializer=tf.keras.initializers.GlorotNormal()):
    """residual feature aggregate block"""
    identity = x
    out1 = dense_block(
        x, kernel_initializer=kernel_initializer, residual_scaling=1)
    x = add([x, out1])
    out2 = dense_block(
        x, kernel_initializer=kernel_initializer, residual_scaling=1)
    x = add([x, out2])
    out3 = dense_block(
        x, kernel_initializer=kernel_initializer, residual_scaling=1)
    x = concatenate([out1, out2, out3])
    x = Conv2D(filters=64,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               kernel_initializer=kernel_initializer, )(x)
    return add([x, identity])


def rfa(x, kernel_initializer=tf.keras.initializers.GlorotNormal()):
    """residual feature aggregate block"""
    identity = x
    out1 = dense_block(
        x, kernel_initializer=kernel_initializer, residual_scaling=1)
    x = add([x, out1])
    out2 = dense_block(
        x, kernel_initializer=kernel_initializer, residual_scaling=1)
    x = add([x, out2])
    out3 = dense_block(
        x, kernel_initializer=kernel_initializer, residual_scaling=1)
    x = add([x, out3])
    out4 = dense_block(
        x, kernel_initializer=kernel_initializer, residual_scaling=1)
    x = concatenate([out1, out2, out3, out4])
    x = Conv2D(filters=64,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               kernel_initializer=kernel_initializer, )(x)
    return add([x, identity])
