import tensorflow as tf
import numpy as np


def read_and_decode(filename_queue, shuffle_batch = True, batch_size = 4, capacity = 4000, num_threads = 4, min_after_dequeue = 2000, const_height = 128, const_width = 256):
    
    reader = tf.TFRecordReader()

    _, serialized_example= reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        }
    )

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.uint8)

    img_shape = tf.stack([height, width, 3])
    img = tf.reshape(img, img_shape)

    resized_img = tf.image.resize_images(img, (const_height, const_width))

    img = tf.image.resize_images(img, [height, width])
    # print('hehe1')
    if shuffle_batch:
        imgs, labels = tf.train.shuffle_batch([resized_img, label], batch_size = batch_size, capacity = capacity, num_threads = num_threads, min_after_dequeue = min_after_dequeue)
    else:
        imgs, labels = tf.train.batch([resized_img, label], batch_size = batch_size, capacity = capacity, num_threads = num_threads)
    # print('hehe2')
    return imgs, labels
