import tensorflow as tf
from tensorflow.keras.layers import Conv2D, add, concatenate, LeakyReLU


def receptive_field_block(x, input_channels: int, output_channels: int,
                          kernel_initializer=tf.keras.initializers.GlorotNormal()):
    branch_channels = input_channels // 4
    # shortcut branch
    short_cut = Conv2D(filters=output_channels,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='same',
                       kernel_initializer=kernel_initializer, )(x)
    short_cut = short_cut * 0.2
    # branch 1
    branch1 = Conv2D(filters=branch_channels,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer, )(x)
    branch1 = LeakyReLU(alpha=0.2)(branch1)
    branch1 = Conv2D(filters=branch_channels,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     dilation_rate=(1, 1), )(branch1)
    # branch 2
    branch2 = Conv2D(filters=branch_channels,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer, )(x)
    branch2 = LeakyReLU(alpha=0.2)(branch2)
    branch2 = Conv2D(filters=branch_channels,
                     kernel_size=(1, 3),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer, )(branch2)
    branch2 = LeakyReLU(alpha=0.2)(branch2)
    branch2 = Conv2D(filters=branch_channels,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     dilation_rate=(3, 3))(branch2)

    # branch 3
    branch3 = Conv2D(filters=branch_channels,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer, )(x)
    branch3 = LeakyReLU(alpha=0.2)(branch3)
    branch3 = Conv2D(filters=branch_channels,
                     kernel_size=(3, 1),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer, )(branch3)
    branch3 = LeakyReLU(alpha=0.2)(branch3)
    branch3 = Conv2D(filters=branch_channels,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     dilation_rate=(3, 3), )(branch3)
    # branch4
    branch4 = Conv2D(filters=branch_channels // 2,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer, )(x)
    branch4 = LeakyReLU(alpha=0.2)(branch4)
    branch4 = Conv2D(filters=(branch_channels // 4) * 3,
                     kernel_size=(1, 3),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer, )(branch4)
    branch4 = LeakyReLU(alpha=0.2)(branch4)
    branch4 = Conv2D(filters=branch_channels,
                     kernel_size=(3, 1),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer, )(branch4)
    branch4 = LeakyReLU(alpha=0.2)(branch4)
    branch4 = Conv2D(filters=branch_channels,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     dilation_rate=(5, 5), )(branch4)
    # concat
    x = concatenate([branch1, branch2, branch3, branch4])
    # 1*1 conv
    x = Conv2D(filters=output_channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               kernel_initializer=kernel_initializer, )(x)
    # add
    return add([x, short_cut])
