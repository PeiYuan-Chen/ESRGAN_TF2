import tensorflow as tf
from utils.postprocess import post_process
from utils.color_utils import rgb_to_ycbcr_y


def calculate_psnr(y_true, y_pred, scale, y_only=True):
    """
    :param y_true: image batch    :param y_pred: image batch
    :param scale: upscale factor
    :param y_only:
    :return:
    """
    # post process
    y_true = post_process(y_true)
    y_pred = post_process(y_pred)
    # crop edge
    boundarypixels = 6 + scale
    y_true = y_true[:, boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
    y_pred = y_pred[:, boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
    # convert to y
    if y_only:
        y_true = rgb_to_ycbcr_y(y_true)
        y_pred = rgb_to_ycbcr_y(y_pred)
    # calculate psnr
    batch_psnr = tf.image.psnr(y_true, y_pred, max_val=255)
    return tf.reduce_mean(batch_psnr)


def calculate_ssim(y_true, y_pred, scale, y_only=True):
    """
    :param y_true: image batch    :param y_pred: image batch
    :param scale: upscale factor
    :param y_only:
    :return:
    """
    # post process
    y_true = post_process(y_true)
    y_pred = post_process(y_pred)
    # crop edge
    boundarypixels = 6 + scale
    y_true = y_true[:, boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
    y_pred = y_pred[:, boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
    # convert to y
    if y_only:
        y_true = rgb_to_ycbcr_y(y_true)
        y_pred = rgb_to_ycbcr_y(y_pred)

    # calculate psnr
    batch_ssim = tf.image.ssim(y_true, y_pred, max_val=255)
    return tf.reduce_mean(batch_ssim)
