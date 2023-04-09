import tensorflow as tf


def post_process(prediction):
    # 确保输出值在[0, 255]范围内
    clipped_prediction = tf.clip_by_value(prediction, 0, 255)
    # 将输出类型从float32转换为uint8
    uint8_prediction = tf.cast(clipped_prediction, tf.uint8)
    return uint8_prediction