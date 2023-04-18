import tensorflow as tf
from configs.load_gan_config import cfg
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import os
import matplotlib.pyplot as plt
from utils.gradient_map import gradient_intensity_map
# load test data
hr_dir = cfg.test_hr_dir
hr_img_paths = sorted(os.listdir(hr_dir))

for hr_path in hr_img_paths:
    hr_img = img_to_array(load_img(os.path.join(hr_dir, hr_path)))
    hr_img = tf.expand_dims(hr_img, axis=0)
    grad_img = gradient_intensity_map(hr_img)
    hr_img = tf.squeeze(hr_img, axis=0)
    grad_img = tf.squeeze(grad_img, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    axes[0].imshow(array_to_img(hr_img))
    axes[1].imshow(array_to_img(grad_img))
    for ax in axes:
        ax.axis('off')

    plt.show()
