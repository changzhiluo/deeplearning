import tensorflow as tf
import numpy as np
import os
from read_img_from_tfrecord import read_and_decode

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tfrecord_filename = 'Data/hehe.train.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
    with tf.Session() as sess:
        sess.run(init_op)
        imgs, labels = read_and_decode(filename_queue, shuffle_batch = True, \
                                                            batch_size = 4, capacity = 4000, \
                                                            num_threads = 4, min_after_dequeue = 2000, \
                                                            const_height = 128, const_width = 256)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        try:
            for ind in range(6):
                img, label = sess.run([imgs, labels]) #一次取出一个batch
                print('img.shape is:', img.shape)
                print('label is: ', label)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)

