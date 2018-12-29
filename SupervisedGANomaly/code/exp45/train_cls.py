#---------------------------------------------------------
# Function: Conduct training of the defect detection model
# Author:   Changzhi Luo
# Date:     20180908
#---------------------------------------------------------
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import os
import sys
from networks import gnet, dnet, cnet
from data_io import load_data, random_flip, convert_to_one_hot, load_data_from_img_list
from config import param
import logging
import json

# Define log format
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'
DATE_FORMAT = '%m/%d/%Y %H:%M:%S %p'
train_log_dir = param['train_log_dir']
if not os.path.exists(train_log_dir):
    os.makedirs(train_log_dir)
logging.basicConfig(filename = os.path.join(train_log_dir, 'DUNet.log'), level = logging.DEBUG, format = LOG_FORMAT, datefmt = DATE_FORMAT)

if __name__ == '__main__':
    # Load training data
    # train_ok_dir = param['image_dir'] + '/train/ok'
    # train_ok_img = load_data(train_ok_dir, param['load_width'], param['load_height'], param['pixel_means'])
    # num_ok = train_ok_img.shape[0]
    
    # train_ng_dir = param['image_dir'] + '/train/ng'
    # train_ng_img = load_data(train_ng_dir, param['load_width'], param['load_height'], param['pixel_means'])
    # num_ng = train_ng_img.shape[0]

    # Load training data list
    with open('data_file/tr_ok_path.json', 'r') as f:
        tr_ok_read_dict = json.load(f)
    tr_ok_id_list = tr_ok_read_dict['im_id_list']
    tr_ok_path_list = tr_ok_read_dict['im_path_list']
    num_ok = len(tr_ok_id_list)

    with open('data_file/tr_ng_path.json', 'r') as f:
        tr_ng_read_dict = json.load(f)
    tr_ng_id_list = tr_ng_read_dict['im_id_list']
    tr_ng_path_list = tr_ng_read_dict['im_path_list']
    num_ng = len(tr_ng_id_list)

    # Define training data list, which needs to be shuffled
    #ok_list = list(range(num_ok))
    #ng_list = list(range(num_ng))
    batch_size = param['batch_size']
    

    # Define the training graph
    alter_graph = tf.Graph()
    with alter_graph.as_default():
        # define G loss
        input_data_ok_ng = tf.placeholder(tf.float32, shape = [batch_size, param['load_height'], param['load_width'], 3], name = 'input_data_ok_ng') # batch_size * 256 * 448 * 3
        input_label_ok_ng = tf.placeholder(tf.float32, shape = [batch_size], name = 'input_label_ok_ng') # batch_size
        input_label_ok_ng_cls = tf.placeholder(tf.float32, shape = [batch_size, 2], name = 'input_label_ok_ng_cls') # batch_size * 2, for softmax Cross entropy
        
        input_csw_ok = tf.placeholder(tf.float32, shape = (), name = 'input_csw_ok') 
        input_csw_ng = tf.placeholder(tf.float32, shape = (), name = 'input_csw_ng') 

        #为了兼容后面的输入label，将int改为float类型
        #input_label_ok_ng = tf.placeholder(tf.float32, shape = [batch_size], name = 'input_label_ok_ng') # batch_size
        
        ok_ind = tf.where(tf.equal(input_label_ok_ng, 1))
        ng_ind = tf.where(tf.equal(input_label_ok_ng, 0))
        input_data_ok_real = tf.gather_nd(input_data_ok_ng, ok_ind) # 3/4 batchsize * 256 * 448 * 3
        #input_data_ng_real = tf.gather_nd(input_data_ok_ng, ng_ind)
        
        input_label_ok_fake = tf.placeholder(tf.int32, shape = [None], name = 'input_label_ok_fake') # ok image per batch (e.g. 3/4 batch_size)

        bottle1_ok, bottle2_ok, input_data_ok_fake = gnet(input_data_ok_real) # feed forward all image, but only use ok image to train gnet and dnet
        output_label_ok_fake = dnet(input_data_ok_fake)

        input_data_D = tf.concat([input_data_ok_real, input_data_ok_fake], 0)
        input_label_D = tf.placeholder(tf.int32, shape = input_data_D.shape[0], name = 'input_label_D') 
        output_label_D = dnet(input_data_D) # 64 * 1

        #input_data_ok_ng = tf.concat([input_data_ok_real, input_data_ng_real], 0)
        bottle1_ok_ng, bottle2_ok_ng, _ = gnet(input_data_ok_ng)
        cls_result = cnet(bottle1_ok_ng, bottle2_ok_ng) # batch_size * 2
        #print("cls_result shape is:", cls_result.shape)

        rec_loss = param['weight_R'] * tf.losses.absolute_difference(input_data_ok_real, input_data_ok_fake)
        enc_loss = param['weight_E'] * tf.losses.mean_squared_error(bottle1_ok, bottle2_ok)
        dloss_fake = param['weight_D'] * tf.losses.log_loss(input_label_ok_fake, output_label_ok_fake)
        
        dnet_loss = param['weight_D'] * tf.losses.log_loss(input_label_D, output_label_D)
        gnet_loss = rec_loss + enc_loss + dloss_fake
        #input_label_ok_ng_onehot = tf.one_hot(input_label_ok_ng, 2)
        #input_label_ok_ng_onehot_stop = tf.stop_gradient(input_label_ok_ng_onehot)
        input_label_ok_ng_cls_stop = tf.stop_gradient(input_label_ok_ng_cls)
        input_label_ok_cls = tf.gather_nd(input_label_ok_ng_cls, ok_ind) 
        input_label_ng_cls = tf.gather_nd(input_label_ok_ng_cls, ng_ind) 
        cls_result_ok = tf.gather_nd(cls_result, ok_ind) 
        cls_result_ng = tf.gather_nd(cls_result, ng_ind) 
        #cnet_loss_softmax = param['weight_C'] * tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_label_ok_ng_cls_stop, logits=cls_result)
        #cnet_loss = tf.reduce_mean(cnet_loss_softmax)
        cnet_loss_softmax_ok = param['weight_C'] * tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_label_ok_cls, logits=cls_result_ok)
        cnet_loss_ok = tf.reduce_mean(cnet_loss_softmax_ok)
        cnet_loss_softmax_ng = param['weight_C'] * tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_label_ng_cls, logits=cls_result_ng)
        cnet_loss_ng = tf.reduce_mean(cnet_loss_softmax_ng)
        cnet_loss = input_csw_ok * cnet_loss_ok + input_csw_ng * cnet_loss_ng


        summary_op1 = tf.summary.scalar('losses/rec_loss', rec_loss)
        summary_op2 = tf.summary.scalar('losses/enc_loss', enc_loss)
        summary_op3 = tf.summary.scalar('losses/dloss_fake', dloss_fake)
        summary_op4 = tf.summary.scalar('losses/dnet_loss', dnet_loss)
        summary_op5 = tf.summary.scalar('losses/gnet_loss', gnet_loss)
        summary_op6 = tf.summary.scalar('losses/cnet_loss', cnet_loss)

        # Specify the update mode of the learning rate
        train_init_lr = tf.Variable(tf.constant(0.))
        train_global_step = tf.Variable(tf.constant(0))
        train_decay_step = tf.Variable(tf.constant(0))
        train_decay_rate = tf.Variable(tf.constant(0.)) #should be float
        train_learning_rate =  tf.train.exponential_decay(learning_rate = train_init_lr, \
                                                            global_step = train_global_step, \
                                                            decay_steps = train_decay_step, \
                                                            decay_rate = train_decay_rate)
        # Specify the optimization scheme
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = train_learning_rate) # try adam later

        var_list_dnet = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnet_scope')
        var_list_gnet = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gnet_scope')
        var_list_cnet = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnet_scope')
        
        dnet_train_op = optimizer.minimize(dnet_loss, var_list=var_list_dnet)
        gnet_train_op = optimizer.minimize(gnet_loss, var_list=var_list_gnet)
        cnet_train_op = optimizer.minimize(cnet_loss, var_list=var_list_cnet)

        # Define saver
        saver = tf.train.Saver(max_to_keep = param['max_to_keep'])

    # Config session
    os.environ['CUDA_VISIBLE_DEVICES'] = param['gpu_num']
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = param['gpu_memory_fraction'])
    tfconfig = tf.ConfigProto(allow_soft_placement = False, gpu_options = gpu_options)

    with tf.Session(graph = alter_graph, config = tfconfig) as sess:
        # Set summary information directory
        tb_log_dir = param['tensorboard_log_dir']
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)
        writer = tf.summary.FileWriter(tb_log_dir, sess.graph)

        # Initilize all variables
        sess.run(tf.global_variables_initializer())

        # Set learning rate ralated info
        init_lr = param['init_lr']
        stepsize = param['stepsize']
        decay_rate = param['decay_rate']


        # Perform training
        print('Start training...')
        for i in range(param['max_iters']):
            # Set training images and labels of each mini-batch

            random.shuffle(tr_ok_id_list)
            random.shuffle(tr_ng_id_list)
            # oversampling strategy, ng_ratio: 1/4 ~ 1/2
            ng_num_per_batch = np.random.randint(int(param['ng_ratio_low'] * batch_size), int(param['ng_ratio_high'] * batch_size) + 1)
            ok_num_per_batch = batch_size - ng_num_per_batch
            # get cost sensitive weight
            csw_ok = float(batch_size) / ok_num_per_batch
            csw_ng = float(batch_size) / ng_num_per_batch

            batch_label_ok_real_fake = np.zeros([ok_num_per_batch * 2])
            batch_label_ok_real_fake[0:ok_num_per_batch] = 1 # ok
            batch_label_ok_real_fake[ok_num_per_batch:] = 0 # ng
            
            batch_label_ok_fake = np.zeros([ok_num_per_batch])

            #batch_img_ok = train_ok_img[tr_ok_id_list[0:ok_num_per_batch], :, :, :] # 3*4/batch_size
            #batch_img_ng = train_ng_img[tr_ng_id_list[0:ng_num_per_batch], :, :, :] # 1*4/batch_size
            ok_img_list = [tr_ok_path_list[j] for j in tr_ok_id_list[0:ok_num_per_batch]]
            ng_img_list = [tr_ng_path_list[j] for j in tr_ng_id_list[0:ng_num_per_batch]]
            batch_img_ok = load_data_from_img_list(ok_img_list, param['load_width'], param['load_height'], param['pixel_means'])
            batch_img_ng = load_data_from_img_list(ng_img_list, param['load_width'], param['load_height'], param['pixel_means'])
            batch_img = np.concatenate((batch_img_ok, batch_img_ng), axis = 0) # batch_size

            batch_label_ok = np.ones([ok_num_per_batch]) # 3*4/batch_size
            batch_label_ng = np.zeros([ng_num_per_batch]) # 1*4/batch_size
            batch_label = np.concatenate((batch_label_ok, batch_label_ng)) # batch_size

            batch_label_cls = convert_to_one_hot(batch_label.astype(int), 2)
 
            ## random flip augmentation
            # batch_img = random_flip(batch_img)

            # One-step training 
            current_global_step = int(i / stepsize) * stepsize

            # Alternatively training D, G, C (first D, then G, then C)
            dnet_loss_scalar, lr, _ = sess.run([dnet_loss, train_learning_rate, dnet_train_op], \
                                    feed_dict = {input_data_ok_ng: batch_img, \
                                                input_label_ok_ng: batch_label, \
                                                input_label_D: batch_label_ok_real_fake, \
                                                train_init_lr: init_lr, \
                                                train_global_step: current_global_step, \
                                                train_decay_step: stepsize, \
                                                train_decay_rate: decay_rate}) # train_learning_rate is the final value

            rec_loss_scalar, enc_loss_scalar, dloss_fake_scalar, gnet_loss_scalar, lr, _ = sess.run([rec_loss, enc_loss, \
                                    dloss_fake, gnet_loss, train_learning_rate, gnet_train_op], \
                                    feed_dict = {input_data_ok_ng: batch_img, \
                                                input_label_ok_ng: batch_label, \
                                                input_label_ok_fake: batch_label_ok_fake, \
                                                train_init_lr: init_lr, \
                                                train_global_step: current_global_step, \
                                                train_decay_step: stepsize, \
                                                train_decay_rate: decay_rate}) # train_learning_rate is the final value

            cnet_loss_scalar, lr, _ = sess.run([cnet_loss, train_learning_rate, cnet_train_op], \
                                    feed_dict = {input_data_ok_ng: batch_img, \
                                                input_label_ok_ng: batch_label, \
                                                input_label_ok_ng_cls: batch_label_cls, \
                                                input_csw_ok: csw_ok, \
                                                input_csw_ng: csw_ng, \
                                                train_init_lr: init_lr, \
                                                train_global_step: current_global_step, \
                                                train_decay_step: stepsize, \
                                                train_decay_rate: decay_rate}) # train_learning_rate is the final value

            # Display training info
            if (i + 1) % param['summary_interval'] == 0: # Note that i starts from 0

                rec_loss_val = sess.run(summary_op1, feed_dict = {input_data_ok_ng: batch_img, input_label_ok_ng: batch_label}) # tensor
                enc_loss_val = sess.run(summary_op2, feed_dict = {input_data_ok_ng: batch_img, input_label_ok_ng: batch_label})
                dloss_fake_val = sess.run(summary_op3, feed_dict = {input_data_ok_ng: batch_img, input_label_ok_ng: batch_label, input_label_ok_fake: batch_label_ok_fake})
                dnet_loss_val = sess.run(summary_op4, feed_dict = {input_data_ok_ng: batch_img, input_label_ok_ng: batch_label, input_label_D: batch_label_ok_real_fake})
                gnet_loss_val = sess.run(summary_op5, feed_dict = {input_data_ok_ng: batch_img, input_label_ok_ng: batch_label, input_label_ok_fake: batch_label_ok_fake})
                cnet_loss_val = sess.run(summary_op6, feed_dict = {input_data_ok_ng: batch_img, input_label_ok_ng: batch_label, input_label_ok_ng_cls: batch_label_cls, \
                                                                    input_csw_ok: csw_ok, input_csw_ng: csw_ng})
                writer.add_summary(rec_loss_val, i + 1)
                writer.add_summary(enc_loss_val, i + 1)
                writer.add_summary(dloss_fake_val, i + 1)
                writer.add_summary(dnet_loss_val, i + 1)
                writer.add_summary(gnet_loss_val, i + 1)
                writer.add_summary(cnet_loss_val, i + 1)
                #logging.info('Iter %d, learning rate is %f, rloss is %f, dloss is %f.', i + 1, lr, gnet_loss, dnet_loss)
                #print('Iter %d, learning rate is %f, rloss is %f, dloss is %f.'%(i + 1, lr, gnet_loss, dnet_loss))
                logging.info('Iter %d, lr is %f, dloss is %f, gloss is %f, closs is %f.', i + 1, lr, dnet_loss_scalar, gnet_loss_scalar, cnet_loss_scalar)
                print('Iter %d, lr is %f, dloss is %f, gloss is %f, closs is %f.'%(i + 1, lr, dnet_loss_scalar, gnet_loss_scalar, cnet_loss_scalar))
 
            # Save model
            if (i + 1) % param['snap_interval'] == 0:
                saver.save(sess, param['model_dir'] + '/dunet_' + str(i + 1) + '.ckpt')
                logging.info('Save model at iter %d.', i + 1)



