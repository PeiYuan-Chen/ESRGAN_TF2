import os
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
from train_utils.metrics import calculate_psnr, calculate_ssim
from configs.load_psnr_config import cfg
from models.model_builder import generator


def eval():
    # load model
    model = generator()
    model.load_weights(cfg.best_weights_file)

    # load eval data
    lr_dir = cfg.eval_lr_dir
    hr_dir = cfg.eval_hr_dir
    lr_img_paths = sorted(os.listdir(lr_dir))
    hr_img_paths = sorted(os.listdir(hr_dir))
    total_psnr = 0.0
    total_ssim = 0.0
    num = 0
    for lr_path, hr_path in zip(lr_img_paths, hr_img_paths):
        lr_img = img_to_array(load_img(os.path.join(lr_dir, lr_path)))
        hr_img = img_to_array(load_img(os.path.join(hr_dir, hr_path)))
        lr_img = tf.expand_dims(lr_img, axis=0)
        hr_img = tf.expand_dims(hr_img, axis=0)
        sr_img = model(lr_img, training=False)
        cur_psnr = calculate_psnr(
            y_true=hr_img, y_pred=sr_img, scale=cfg.upscale_factor, y_only=False)
        cur_ssim = calculate_ssim(
            y_true=hr_img, y_pred=sr_img, scale=cfg.upscale_factor, y_only=False)
        total_psnr += cur_psnr
        total_ssim += cur_ssim
        num += 1

    mean_psnr = total_psnr / num
    mean_ssim = total_ssim / num

    with open(cfg.eval_log_file, "w") as f:
        f.write(f'Mean PSNR: {mean_psnr}\n')
        f.write(f'Mean SSIM: {mean_ssim}\n')

    print(f'Mean PSNR: {mean_psnr}')
    print(f'Mean SSIM: {mean_ssim}')


if __name__ == '__main__':
    eval()
