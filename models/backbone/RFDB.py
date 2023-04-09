import tensorflow as tf
from tensorflow.keras.layers import add, LeakyReLU, concatenate
from models.backbone.RFB import receptive_field_block


def receptive_field_dense_block(x, channels=64, growth_channels=32, residual_scaling=0.2,
                                kernel_initializer=tf.keras.initializers.GlorotNormal()):
    identity = x
    out1 = receptive_field_block(x,
                                 input_channels=channels + 0 * growth_channels,
                                 output_channels=growth_channels,
                                 kernel_initializer=kernel_initializer)
    out1 = LeakyReLU(alpha=residual_scaling)(out1)
    out2 = receptive_field_block(concatenate([x, out1], axis=-1),
                                 input_channels=channels + 1 * growth_channels,
                                 output_channels=growth_channels,
                                 kernel_initializer=kernel_initializer)
    out2 = LeakyReLU(alpha=residual_scaling)(out2)
    out3 = receptive_field_block(concatenate([x, out1, out2], axis=-1),
                                 input_channels=channels + 2 * growth_channels,
                                 output_channels=growth_channels,
                                 kernel_initializer=kernel_initializer)
    out3 = LeakyReLU(alpha=residual_scaling)(out3)
    out4 = receptive_field_block(concatenate([x, out1, out2, out3], axis=-1),
                                 input_channels=channels + 3 * growth_channels,
                                 output_channels=growth_channels,
                                 kernel_initializer=kernel_initializer)
    out4 = LeakyReLU(alpha=residual_scaling)(out4)
    out5 = receptive_field_block(concatenate([x, out1, out2, out3, out4], axis=-1),
                                 input_channels=channels + 4 * growth_channels,
                                 output_channels=channels,
                                 kernel_initializer=kernel_initializer)
    out = out5 * residual_scaling
    return add([out, identity])