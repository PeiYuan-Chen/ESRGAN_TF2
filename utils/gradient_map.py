import tensorflow as tf


def gradient_intensity_map(images):
    """convert batch RGB images to grayscale and then calculate its gradient intensity map"""
    images = tf.cast(images, dtype=tf.float32)
    # Convert the images to grayscale since  they're RGB
    images = tf.image.rgb_to_grayscale(images)

    # Define Sobel kernels
    sobel_x = tf.constant([[0, 0, 0],
                           [-1, 0, 1],
                           [0, 0, 0]], dtype=tf.float32)
    sobel_y = tf.constant([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]], dtype=tf.float32)

    # Reshape the kernels for conv2d
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

    # Apply convolution to the images using Sobel kernels
    grad_x = tf.nn.conv2d(images, sobel_x, strides=[
                          1, 1, 1, 1], padding='SAME')
    grad_y = tf.nn.conv2d(images, sobel_y, strides=[
                          1, 1, 1, 1], padding='SAME')

    # Calculate the gradient magnitude
    gradient_magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y) + 1e-6)

    return gradient_magnitude
