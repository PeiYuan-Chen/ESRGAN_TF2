import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, Softmax, add, PReLU, LayerNormalization


def in_scale_non_local_attention(input_tensor, channel_reduction=2, softmax_factor=6):
    # input_tensor (b,h,w,c)
    batch_size, height, width, channels = input_tensor.shape
    inter_channels = channels // channel_reduction

    theta = Conv2D(filters=inter_channels,
                   kernel_size=(1, 1),
                   strides=1,
                   padding='same',)(input_tensor)
    theta = LayerNormalization()(theta)  # Normalize along the channel
    # activation
    phi = Conv2D(filters=inter_channels,
                 kernel_size=(1, 1),
                 strides=1,
                 padding='same',)(input_tensor)
    phi = LayerNormalization()(phi)  # Normalize along the channel

    # activation
    g = Conv2D(filters=inter_channels,
               kernel_size=(1, 1),
               strides=1,
               padding='same',)(input_tensor)
    g = LayerNormalization()(g)  # Normalize along the channel
    # activation
    # theta,phi,g (b,h,w,c//2)
    theta_flat = tf.reshape(tensor=theta,
                            shape=(batch_size, height*width, inter_channels))
    phi_flat = tf.reshape(tensor=phi,
                          shape=(batch_size, height*width, inter_channels))
    g_flat = tf.reshape(tensor=g,
                        shape=(batch_size, height*width, inter_channels))
    # theta_flat, phi_flat g_flat (b,h*w,c//2)
    attention_map = tf.matmul(theta_flat, phi_flat,
                              transpose_b=True)  # (b,h*w,h*w)
    attention_map_softmax = tf.nn.softmax(
        attention_map*softmax_factor, axis=-1)

    y = tf.matmul(attention_map_softmax, g_flat)  # (b,h*w,c//2)
    y = tf.reshape(tensor=y,
                   shape=(batch_size, height, width, inter_channels))
    y = Conv2D(filters=channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same')(y)
    return y


def cross_scale_non_local_attention(input_tensor, channel_reduction=2, scale=3, patch_size=3, softmax_scale=10):
    batch_size, height, width, channels = input_tensor.shape
    inter_channels = channels // channel_reduction
    inter_height = height // scale
    inter_width = width // scale

    g = Conv2D(filters=channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same')(input_tensor)
    g = PReLU()(g)  # (b,h,w,c)

    g_patch = tf.image.extract_patches(images=g,
                                       sizes=(1, scale*patch_size,
                                              scale*patch_size, 1),
                                       strides=(1, scale, scale, 1),
                                       padding='same')  # (b,h/s,w/s,s*p*s*p*c)
    g_patch = tf.reshape(tensor=g_patch,
                         shape=(batch_size, inter_height*inter_width, scale*patch_size, scale*patch_size, channels))
    # (b,N,s*p,s*p,c) N = hw/(s*s)

    theta = Conv2D(filters=inter_channels,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='same',)(input_tensor)
    theta = PReLU()(theta)  # (b,h,w,c/2)

    phi = tf.image.resize(input_tensor, size=(
        inter_height, inter_width), method=tf.image.ResizeMethod.BILINEAR)
    phi = Conv2D(filters=inter_channels,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 padding='same')(phi)
    phi = PReLU()(phi)  # (b,h/s,w/s,c/2)

    phi_patch = tf.image.extract_patches(images=phi,
                                         sizes=(1, patch_size, patch_size, 1),
                                         strides=(1, 1, 1, 1),
                                         rates=(1, 1, 1, 1),
                                         padding='same')  # (b,h/s,w/s,p*p*c/2)
    phi_patch = tf.reshape(tensor=phi_patch,
                           shape=(batch_size, inter_height*inter_width, patch_size, patch_size, inter_channels))
    # (b,N,p,p,c/2) N = hw/(s*s)

    phi_patch = tf.unstack(phi_patch, axis=0)
    g_patch = tf.unstack(g_patch, axis=0)
    theta = tf.split(theta, num_or_size_splits=batch_size, axis=0)

    y = []

    for theta_i, phi_patch_i, g_patch_i in zip(theta, phi_patch, g_patch):
        # theta_i (1,h,w,c/2) split
        # phi_patch_i (N,p,p,c/2) unstack
        # g_patch_i (N,s*p,s*p,c) unstack
        # normalize phi_patch_i
        max_phi_patch_i = tf.sqrt(tf.reduce_sum(
            tf.square(phi_patch_i), axis=[1, 2, 3], keepdims=True))
        max_phi_patch_i = tf.maximum(max_phi_patch_i, tf.fill(
            max_phi_patch_i.shape, max_phi_patch_i))
        phi_patch_i = phi_patch_i / max_phi_patch_i

        phi_patch_i = tf.transpose(phi_patch_i,
                                   perm=(1, 2, 3, 0))  # (p,p,c/2,N)

        y_i = tf.nn.conv2d(input=theta_i,
                           filters=phi_patch_i,
                           strides=(1, 1),
                           padding='same',
                           data_format='NHWC')  # (1,h,w,N)
        y_i_softmax = tf.nn.softmax(y_i*softmax_scale, axis=-1)  # feature map

        # g_patch_i (s*p,s*p,c,N)
        g_patch_i = tf.transpose(g_patch_i, perm=[1, 2, 3, 0])
        y_i = tf.nn.conv2d_transpose(input=y_i_softmax,
                                     filters=g_patch_i,
                                     output_shape=(
                                         1, scale*height, scale*width, channels),
                                     strides=scale,
                                     padding='SAME',
                                     data_format='NHWC')  # (1,s*h,s*w,c)
        y_i = y_i / 6
        y.append(y_i)
    y = tf.concat(y, axis=0)
    return y
