import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

img_dir = './Data/hehe/train'
tfrecord_filename = 'Data/hehe.train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecord_filename)

#height = 128
#width = 256

name_list = os.listdir(img_dir)
for ind, img_name in enumerate(name_list):
    img_path = os.path.join(img_dir, img_name)
    img = np.array(Image.open(img_path))
    # img_raw = img_raw.tobytes() # 将图片转化为二进制格式
    height = img.shape[0]
    width = img.shape[1]
    label = 1
    img_raw = img.tostring()

    example = tf.train.Example(features = tf.train.Features(feature = {
        #不加encode()会报错‘TypeError: '000664.jpg' has type str, but expected one of: bytes’
        'height': int64_feature(height),
        'width': int64_feature(width),
        'img_raw': bytes_feature(img_raw),
        'label': int64_feature(label)
    }))

    writer.write(example.SerializeToString())

writer.close()




