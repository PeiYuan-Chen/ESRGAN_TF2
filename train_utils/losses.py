import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy


def _vgg(output_layer):
    vgg = VGG19(weights='imagenet', input_shape=(
        None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def make_perceptual_loss(criterion='l1', output='54', before_act=True):
    """loss type"""
    if criterion == 'l1':
        loss_fn = MeanAbsoluteError()
    elif criterion == 'l2':
        loss_fn = MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))

    """output layer"""
    vgg = VGG19(weights='imagenet', input_shape=(
        None, None, 3), include_top=False)
    if output == '22':
        output_layer = 5
    elif output == '54':
        output_layer = 20
    else:
        raise NotImplementedError(
            'VGG output layer {} is not recognized.'.format(criterion))
    if before_act:
        vgg.layers[output_layer].activation = None
    fea_out = Model(vgg.input, vgg.layers[output_layer].output)

    def perceptual_loss(y_true, y_pred):
        y_true = preprocess_input(tf.cast(y_true,tf.float32)) / 12.75
        y_pred = preprocess_input(y_pred) / 12.75
        return loss_fn(fea_out(y_true), fea_out(y_pred))

    return perceptual_loss


def make_pixel_loss(criterion='l1'):
    if criterion == 'l1':
        return MeanAbsoluteError()
    elif criterion == 'l2':
        return MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))


def make_discriminator_loss(gan_type='ragan'):
    cross_entropy = BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def discriminator_loss_ragan(real_output, fake_output):
        real_loss = cross_entropy(y_true=tf.ones_like(real_output),
                                  y_pred=sigma(real_output - tf.reduce_mean(fake_output)))
        fake_loss = cross_entropy(y_true=tf.zeros_like(fake_output),
                                  y_pred=sigma(fake_output - tf.reduce_mean(real_output)))
        return real_loss + fake_loss

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(y_true=tf.ones_like(
            real_output), y_pred=sigma(real_output))
        fake_loss = cross_entropy(y_true=tf.zeros_like(
            fake_output), y_pred=sigma(fake_output))
        total_loss = real_loss + fake_loss
        return total_loss

    if gan_type == 'ragan':
        return discriminator_loss_ragan
    elif gan_type == 'gan':
        return discriminator_loss
    else:
        raise NotImplementedError(
            'Discriminator loss type {} is not recognized.'.format(gan_type))


def make_generator_loss(gan_type='ragan'):
    cross_entropy = BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def generator_loss_ragan(real_output, fake_output):
        loss1 = cross_entropy(y_true=tf.zeros_like(real_output),
                              y_pred=sigma(real_output - tf.reduce_mean(fake_output)))
        loss2 = cross_entropy(y_true=tf.ones_like(fake_output), y_pred=sigma(
            fake_output - tf.reduce_mean(real_output)))
        return loss1 + loss2

    def generator_loss(fake_output):
        return cross_entropy(y_true=tf.ones_like(fake_output), y_pred=sigma(fake_output))

    if gan_type == 'ragan':
        return generator_loss_ragan
    elif gan_type == 'gan':
        return generator_loss
    else:
        raise NotImplementedError(
            'Generator loss type {} is not recognized.'.format(gan_type))
