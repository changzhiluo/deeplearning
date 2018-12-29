#---------------------------------------------------------
# Function: Defect detection interface 
# Author:   Changzhi Luo
# Date:     20180914
#---------------------------------------------------------
import tensorflow as tf 
import numpy as np
# from networks import gnet, dnet, cnet
from networks import gnet, dnet, cnet
from data_io import load_data, write_image
from config import param
import os

class IF_DUNET:
    def __init__(self, model_path = None):
        # 功    能：构造函数，类实例化时调用
        # 入    参：模型路径
        # 返 回 值：无
        # 示    例：if_dunet = IF_DUNET('../model/defect1/dunet_15000.ckpt')
        self.model = model_path
        self.sess = None
        self.graph = None
        self.input_data = None
        self.output_label = None
        self.output_image = None
        self.fea_before_enc = None
        self.fea_after_enc = None

    def init_model(self, input_shape = None, gpu_id = None, allow_gpu_growth = False, per_gpu_fraction = 0.9):
        # 功    能：初始化函数，只在检测任务启动时调用一次，将检测部分的graph和sess载入内存
        # 入    参：gpu配置，网络输入形状
        # 返 回 值：无
        # 示    例：if_dunet.init_dunet_model(input_shape = [1, 256, 448, 3], gpu_id = '0') 

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        gpu_options = tf.GPUOptions(allow_growth = allow_gpu_growth, per_process_gpu_memory_fraction = per_gpu_fraction)
        tfconfig = tf.ConfigProto(allow_soft_placement = False, gpu_options = gpu_options)
        tfgraph = tf.Graph()
        # define computation graph
        with tfgraph.as_default():
            input_data = tf.placeholder(tf.float32, shape = input_shape, name = 'input_data')
            bottle1, bottle2, reconstruct_image = gnet(input_data) # 64 * 256 * 448 * 3
            output_label = cnet(bottle1, bottle2)
            tfsaver = tf.train.Saver()
        tfsess = tf.Session(graph = tfgraph, config = tfconfig)
        tfsaver.restore(tfsess, self.model)
        self.input_data = input_data
        self.output_label = output_label
        self.output_image = reconstruct_image
        self.fea_before_enc = tf.squeeze(bottle1)
        self.fea_after_enc = tf.squeeze(bottle2)
        self.sess = tfsess
        self.graph = tfgraph

    def inference(self, graph, sess, img):
    # 功    能：执行单张图片检测
    # 入    参：待检测图片
    # 返 回 值：图片异常值
    # 示    例：if_dunet.perform_dunet(if_dunet.graph, if_dunet.sess, img)
        graph = self.graph
        sess = self.sess
        output_image_val, fea_before_enc_val, fea_after_enc_val, output_label_val = sess.run([self.output_image, self.fea_before_enc, \
                                                                                              self.fea_after_enc, self.output_label], \
                                                                                              feed_dict = {self.input_data:img})                                                                                              
        return output_image_val, fea_before_enc_val, fea_after_enc_val, output_label_val
