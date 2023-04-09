# train_val.py
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import load_img, img_to_array

from configs.load_psnr_config import cfg
from datasets.dataloader import sr_input_pipline_from_dir, sr_input_pipline_from_tfrecord
from models.model_builder import generator, generator_x4

from train_utils.metrics import calculate_psnr, calculate_ssim
from train_utils.lr_schedules import multistep_lr_schedule
from train_utils.losses import make_pixel_loss
from utils.history import create_or_continue_history, save_history
from train_utils.initializers import scaled_HeNormal


def train():
    # self-define
    model = generator_x4(kernel_initializer=scaled_HeNormal(0.1))
    loss_fn = make_pixel_loss(criterion='l1')

    ###########################
    # no need to modify
    ###########################
    # load data
    if cfg.Use_TFRecord:
        train_ds = sr_input_pipline_from_tfrecord(cfg.TFRecord_file, cfg.cache_dir, cfg.hr_size, cfg.upscale_factor,
                                                  cfg.batch_size, training=True)
    else:
        train_ds = sr_input_pipline_from_dir(cfg.train_lr_dir, cfg.train_hr_dir, cfg.cache_dir, cfg.hr_size, cfg.upscale_factor,
                                             cfg.batch_size, training=True)
    val_lr_dir = cfg.val_lr_dir
    val_hr_dir = cfg.val_hr_dir
    val_lr_img_paths = sorted(os.listdir(val_lr_dir))
    val_hr_img_paths = sorted(os.listdir(val_hr_dir))
    # lr schedule
    lr_schedule = multistep_lr_schedule(initial_lr=cfg.init_learning_rate, lr_decay_iter_list=cfg.lr_decay_iter_list,
                                        lr_decay_rate=cfg.lr_decay_rate)
    # optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    # checkpoint
    latest_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    latest_checkpoint_manager = tf.train.CheckpointManager(
        latest_checkpoint, cfg.latest_checkpoint_dir, max_to_keep=1)

    # Check if the checkpoint directory is not empty
    if os.listdir(cfg.latest_checkpoint_dir):
        # restore latest checkpoint
        the_latest_checkpoint = tf.train.latest_checkpoint(
            cfg.latest_checkpoint_dir)
        print(f'Restoring from latest checkpoint: {the_latest_checkpoint}')
        latest_checkpoint.restore(the_latest_checkpoint)
    else:
        print('No checkpoints found, training from scratch.')

    # restore history
    # the latest history
    history, start_iteration = create_or_continue_history(
        cfg.history_file)
    if start_iteration == 0:
        max_psnr = 0.0
    else:
        max_psnr = history['best_val_psnr']

    total_train_psnr = 0.0
    total_train_ssim = 0.0
    total_train_loss = 0.0

    for i, (x_batch, y_batch) in enumerate(train_ds):
        # iterations
        i += start_iteration
        if i >= cfg.iterations:
            break

        # fit
        with tf.GradientTape() as tape:
            # forward propagation
            y_pred = model(x_batch, training=True)
            # loss
            train_loss = loss_fn(y_true=y_batch, y_pred=y_pred)
        # gradient
        gradient = tape.gradient(train_loss, model.trainable_variables)
        # update
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))

        # train metrics
        train_psnr = calculate_psnr(
            y_true=y_batch, y_pred=y_pred, scale=cfg.upscale_factor, y_only=True)
        train_ssim = calculate_ssim(
            y_true=y_batch, y_pred=y_pred, scale=cfg.upscale_factor, y_only=True)
        total_train_loss += train_loss
        total_train_psnr += train_psnr
        total_train_ssim += train_ssim

        # val_loss
        # every n iterations
        if (i + 1) % cfg.save_every == 0:
            # evaluate metrics in val_ds
            total_val_loss = 0.0
            total_psnr = 0.0
            total_ssim = 0.0
            num = 0
            for lr_path, hr_path in zip(val_lr_img_paths, val_hr_img_paths):
                lr_img = img_to_array(
                    load_img(os.path.join(val_lr_dir, lr_path)))
                hr_img = img_to_array(
                    load_img(os.path.join(val_hr_dir, hr_path)))
                lr_img = tf.expand_dims(lr_img, axis=0)
                hr_img = tf.expand_dims(hr_img, axis=0)
                sr_img = model(lr_img, training=False)
                total_val_loss += loss_fn(y_true=hr_img, y_pred=sr_img)
                cur_psnr = calculate_psnr(
                    y_true=hr_img, y_pred=sr_img, scale=cfg.upscale_factor, y_only=True)
                cur_ssim = calculate_ssim(
                    y_true=hr_img, y_pred=sr_img, scale=cfg.upscale_factor, y_only=True)
                total_psnr += cur_psnr
                total_ssim += cur_ssim
                num += 1
            # calculate mean metrics of train and val
            val_loss = total_val_loss / num
            val_mean_psnr = total_psnr / num
            val_mean_ssim = total_ssim / num
            train_loss = total_train_loss / cfg.save_every
            train_mean_psnr = total_train_psnr / cfg.save_every
            train_mean_ssim = total_train_ssim / cfg.save_every
            # print loss and metrics
            print(f"Iteration {i + 1}, "
                  f"loss: {train_loss}, "
                  f"val_loss: {val_loss}, "
                  f"psnr: {train_mean_psnr}, "
                  f"val_psnr: {val_mean_psnr},"
                  f"ssim: {train_mean_ssim}, "
                  f"val_ssim: {val_mean_ssim}")

            # history
            history['iteration'].append(i + 1)
            history['loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['val_psnr'].append(float(val_mean_psnr))
            history['val_ssim'].append(float(val_mean_ssim))

            # ModelCheckpoint
            latest_checkpoint_manager.save()
            # save history
            save_history(history, cfg.history_file)

            # save best
            if val_mean_psnr > max_psnr:
                max_psnr = val_mean_psnr
                # weight.h5
                model.save_weights(cfg.best_weights_file)
                # save history
                history['best_iteration'] = i + 1
                history['best_val_psnr'] = float(max_psnr)
                save_history(history, cfg.history_file)
                print('save the best')

            # reset
            total_train_loss = 0.0
            total_train_ssim = 0.0
            total_train_psnr = 0.0

    ###########################
    # no need to modify
    ###########################


if __name__ == '__main__':
    sys.setrecursionlimit(2000)
    train()
