"""Data augmentation for lr,hr image pair
"""
import tensorflow as tf


def flip_left_right(lr_img, hr_img):
    """Random(50%) flip image horizontally"""
    rn = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
    return tf.cond(
        rn < 0.5,
        lambda: (lr_img, hr_img),
        lambda: (tf.image.flip_left_right(lr_img), tf.image.flip_left_right(hr_img))
    )


def random_rotate(lr_img, hr_img):
    """Random rotate image for 0,90,180,270 degree"""
    rn = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, k=rn), tf.image.rot90(hr_img, k=rn)


def random_crop(lr_img, hr_img, hr_crop_size, scale):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]  # (height, width)
    lr_height_crop_start = tf.random.uniform(shape=(), minval=0, maxval=lr_img_shape[0] - lr_crop_size + 1,
                                             dtype=tf.int32)
    lr_width_crop_start = tf.random.uniform(shape=(), minval=0, maxval=lr_img_shape[1] - lr_crop_size + 1,
                                            dtype=tf.int32)
    hr_height_crop_start = lr_height_crop_start * scale
    hr_width_crop_start = lr_width_crop_start * scale
    lr_img_cropped = lr_img[
                     lr_height_crop_start:lr_height_crop_start + lr_crop_size,
                     lr_width_crop_start:lr_width_crop_start + lr_crop_size,
                     ]
    hr_img_cropped = hr_img[
                     hr_height_crop_start:hr_height_crop_start + hr_crop_size,
                     hr_width_crop_start:hr_width_crop_start + hr_crop_size,
                     ]
    return lr_img_cropped, hr_img_cropped
