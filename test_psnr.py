import os
import tensorflow as tf
import matplotlib.pyplot as plt
from models.model_builder import generator_x4
from configs.load_psnr_config import cfg
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from utils.postprocess import post_process
from matplotlib.patches import Rectangle


def test():
    # load model
    model = generator_x4()
    model.load_weights(cfg.best_weights_file)
    # zoom region
    y1 = 100
    y2 = 200
    x1 = 100
    x2 = 200
    zoom_region = (slice(y1, y2), slice(x1, x2))
    zoom_rectangle = Rectangle(
        (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
    # load test data
    lr_dir = cfg.test_lr_dir
    hr_dir = cfg.test_hr_dir
    lr_img_paths = sorted(os.listdir(lr_dir))
    hr_img_paths = sorted(os.listdir(hr_dir))

    for lr_path, hr_path in zip(lr_img_paths, hr_img_paths):
        lr_img = img_to_array(load_img(os.path.join(lr_dir, lr_path)))
        hr_img = img_to_array(load_img(os.path.join(hr_dir, hr_path)))
        lr_img = tf.expand_dims(lr_img, axis=0)
        sr_img = model.predict(lr_img)
        sr_img = post_process(sr_img)
        sr_img = tf.squeeze(sr_img, axis=0)
        lr_img = tf.squeeze(lr_img, axis=0)
        lr_img = tf.image.resize(lr_img, size=(
            hr_img.shape[:2]), method='bicubic')

        lr_zoom = lr_img[zoom_region]
        sr_zoom = sr_img[zoom_region]
        hr_zoom = hr_img[zoom_region]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(array_to_img(lr_img))
        axes[0, 0].set_title('Low-resolution image')
        axes[0, 1].imshow(array_to_img(sr_img))
        axes[0, 1].set_title('Super-resolution image')
        axes[0, 2].imshow(array_to_img(hr_img))
        axes[0, 2].set_title('High-resolution image')
        # Create separate rectangle patches for each image in the first row
        lr_zoom_rectangle = Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        sr_zoom_rectangle = Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        hr_zoom_rectangle = Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')

        axes[0, 0].add_patch(lr_zoom_rectangle)
        axes[0, 1].add_patch(sr_zoom_rectangle)
        axes[0, 2].add_patch(hr_zoom_rectangle)

        axes[1, 0].imshow(array_to_img(lr_zoom))
        axes[1, 0].set_title('Zoomed low-resolution image')
        axes[1, 1].imshow(array_to_img(sr_zoom))
        axes[1, 1].set_title('Zoomed super-resolution image')
        axes[1, 2].imshow(array_to_img(hr_zoom))
        axes[1, 2].set_title('Zoomed high-resolution image')

        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')

        plt.show()


if __name__ == '__main__':
    test()

   