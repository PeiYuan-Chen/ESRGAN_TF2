import tensorflow as tf
from datasets.data_augmentation import flip_left_right, random_crop, random_rotate


def get_img_from_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    return img


def load_img_pair_from_dir(lr_dir, hr_dir, cache_file):
    # 1. load image path
    lr_img_ds = tf.data.Dataset.list_files(f'{lr_dir}/*', shuffle=False)
    hr_img_ds = tf.data.Dataset.list_files(f'{hr_dir}/*', shuffle=False)
    # 2. convert path to image
    lr_img_ds = lr_img_ds.map(
        get_img_from_path, num_parallel_calls=tf.data.AUTOTUNE)
    hr_img_ds = hr_img_ds.map(
        get_img_from_path, num_parallel_calls=tf.data.AUTOTUNE)
    # 3. zip to single dataset
    img_ds = tf.data.Dataset.zip((lr_img_ds, hr_img_ds))
    # 4. cache
    img_ds = img_ds.cache(cache_file)
    return img_ds


def parse_tfexample(example_proto):
    feature_description = {
        'lr': tf.io.FixedLenFeature([], tf.string),
        'hr': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    lr_image = tf.image.decode_png(example['lr'], channels=3)
    hr_image = tf.image.decode_png(example['hr'], channels=3)
    return lr_image, hr_image


def load_img_pair_from_tfrecord(record_file, cache_file):
    raw_dataset = tf.data.TFRecordDataset(record_file)
    dataset = raw_dataset.map(
        parse_tfexample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache(cache_file)
    return dataset


def dataset_object(dataset_cache, hr_img_size, scale, batch_size, training=True):
    ds = dataset_cache
    # # 1. shuffle
    # ds = ds.shuffle(buffer_size=len(dataset_cache))
    # 2. repeat
    if training:
        ds = ds.repeat(-1)
    # 4. random crop
    ds = ds.map(
        lambda lr_img, hr_img: random_crop(
            lr_img, hr_img, hr_crop_size=hr_img_size, scale=scale),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # 5. augmentation
    if training:
        ds = ds.map(random_rotate, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=tf.data.AUTOTUNE)
    # 6. batch  7. prefetch
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def sr_input_pipline_from_dir(lr_dir, hr_dir, cache_file, hr_img_size, scale, batch_size, training=True):
    return dataset_object(load_img_pair_from_dir(lr_dir, hr_dir, cache_file), hr_img_size, scale, batch_size, training)


def sr_input_pipline_from_tfrecord(record_file, cache_file, hr_img_size, scale, batch_size, training=True):
    return dataset_object(load_img_pair_from_tfrecord(record_file, cache_file), hr_img_size, scale, batch_size,
                          training)
