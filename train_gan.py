import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import load_img, img_to_array

from configs.load_gan_config import cfg
from datasets.dataloader import sr_input_pipline_from_dir, sr_input_pipline_from_tfrecord
from models.model_builder import generator_x4, discriminator_model_sn

from train_utils.metrics import calculate_psnr, calculate_ssim
from train_utils.lr_schedules import multistep_lr_schedule
from train_utils.losses import make_pixel_loss, make_perceptual_loss, make_generator_loss, make_discriminator_loss
from utils.history import create_or_continue_gan_history, save_history
from train_utils.initializers import scaled_HeNormal


def train_gan():
    generator = generator_x4(kernel_initializer=scaled_HeNormal(0.1))
    discriminator = discriminator_model_sn()
    content_loss_fn = make_pixel_loss(criterion='l1')
    gen_adv_loss_fn = make_generator_loss(gan_type='ragan')
    perc_loss_fn = make_perceptual_loss(
        criterion='l1', output='54', before_act=True)
    dis_adv_loss_fn = make_discriminator_loss(gan_type='ragan')

    # load data
    if cfg.Use_TFRecord:
        train_ds = sr_input_pipline_from_tfrecord(cfg.TFRecord_file, cfg.cache_dir, cfg.hr_size, cfg.upscale_factor,
                                                  cfg.batch_size, training=True)
    else:
        train_ds = sr_input_pipline_from_dir(cfg.train_lr_dir, cfg.train_hr_dir, cfg.cache_dir, cfg.hr_size, cfg.upscale_factor,
                                             cfg.batch_size, training=True)
    # lr schedule
    gen_lr_schedule = multistep_lr_schedule(initial_lr=cfg.gen_init_learning_rate, lr_decay_iter_list=cfg.lr_decay_iter_list,
                                            lr_decay_rate=cfg.lr_decay_rate)
    # optimizer
    gen_optimizer = keras.optimizers.Adam(
        learning_rate=gen_lr_schedule, epsilon=1e-8)

    # lr schedule
    dis_lr_schedule = multistep_lr_schedule(initial_lr=cfg.dis_init_learning_rate, lr_decay_iter_list=cfg.lr_decay_iter_list,
                                            lr_decay_rate=cfg.lr_decay_rate)
    # optimizer
    dis_optimizer = keras.optimizers.Adam(
        learning_rate=dis_lr_schedule, epsilon=1e-8)

    # checkpoint
    latest_checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                            discriminator_optimizer=dis_optimizer,
                                            generator=generator,
                                            discriminator=discriminator)
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
        print('No checkpoints found, training from pretrained generator.')
        generator.load_weights(cfg.gen_pretrained_weight_file)

    history, start_iteration = create_or_continue_gan_history(cfg.history_file)
    total_gen_loss = 0.0
    total_dis_loss = 0.0

    for i, (x_batch, y_batch) in enumerate(train_ds):
        # iterations
        i += start_iteration
        if i >= cfg.iterations:
            break

        # fit
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # forward propagationy
            generator_images = generator(x_batch, training=True)
            real_output = discriminator(y_batch, training=True)
            fake_output = discriminator(generator_images, training=True)
            gen_adv_loss = gen_adv_loss_fn(
                real_output=real_output, fake_output=fake_output)
            disc_loss = dis_adv_loss_fn(
                real_output=real_output, fake_output=fake_output)
            content_loss = content_loss_fn(
                y_true=y_batch, y_pred=generator_images)
            perc_loss = perc_loss_fn(y_true=y_batch, y_pred=generator_images)
            gen_loss = perc_loss + 5e-3 * gen_adv_loss + 1e-2 * content_loss

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        dis_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

        total_gen_loss += gen_loss
        total_dis_loss += disc_loss

        # print
        if (i + 1) % 10 == 0:
            mean_gen_loss = total_gen_loss / cfg.save_every
            mean_disc_loss = total_dis_loss / cfg.save_every
            # print loss and metrics
            print(f"Iteration {i + 1}, "
                  f"gen_loss: {mean_gen_loss}, "
                  f"disc_loss: {mean_disc_loss}, "
                  )
            # history
            history['iteration'].append(i + 1)
            history['gen_loss'].append(float(mean_gen_loss))
            history['disc_loss'].append(float(mean_disc_loss))
            # save history
            save_history(history, cfg.history_file)
            # reset
            total_gen_loss = 0.0
            total_dis_loss = 0.0
        # save n iterations
        if (i + 1) % cfg.save_every == 0:
            # ModelCheckpoint
            latest_checkpoint_manager.save()
            # save weight
            generator.save_weights(cfg.gen_weights_file)
            # print
            print('save weights')

    ###########################
    # no need to modify
    ###########################


if __name__ == '__main__':
    sys.setrecursionlimit(2000)
    train_gan()
