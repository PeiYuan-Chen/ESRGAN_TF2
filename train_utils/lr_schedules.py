import tensorflow as tf


def multistep_lr_schedule(initial_lr, lr_decay_iter_list, lr_decay_rate):
    lr_step_value = [initial_lr]
    for _ in range(len(lr_decay_iter_list)):
        lr_step_value.append(lr_step_value[-1] * lr_decay_rate)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_decay_iter_list, values=lr_step_value)
