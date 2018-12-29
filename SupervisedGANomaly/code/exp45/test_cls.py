#---------------------------------------------------------
# Function: Conduct testing of the defect detection model
#           using interface, use criteria from training ng set
# Author:   Changzhi Luo
# Date:     20180914
#---------------------------------------------------------
import tensorflow as tf 
import numpy as np
from networks import gnet, dnet, cnet
from data_io import load_data, write_image, load_data_from_img_list
from config import param
from interface import IF_DUNET
from evaluation import get_thresh, get_recall_precision
import os
import cv2
import json

if __name__ == '__main__':

    print('Loading data...')
    # Load training data list
    with open('data_file/tr_ok_path.json', 'r') as f:
        tr_ok_read_dict = json.load(f)
    tr_ok_id_list = tr_ok_read_dict['im_id_list']
    tr_ok_path_list = tr_ok_read_dict['im_path_list']
    tr_num_ok = len(tr_ok_id_list)

    with open('data_file/tr_ng_path.json', 'r') as f:
        tr_ng_read_dict = json.load(f)
    tr_ng_id_list = tr_ng_read_dict['im_id_list']
    tr_ng_path_list = tr_ng_read_dict['im_path_list']
    tr_num_ng = len(tr_ng_id_list)
    
    # Load testing data list
    with open('data_file/ts_ok_path.json', 'r') as f:
        ts_ok_read_dict = json.load(f)
    ts_ok_id_list = ts_ok_read_dict['im_id_list']
    ts_ok_path_list = ts_ok_read_dict['im_path_list']
    ts_num_ok = len(ts_ok_id_list)

    with open('data_file/ts_ng_path.json', 'r') as f:
        ts_ng_read_dict = json.load(f)
    ts_ng_id_list = ts_ng_read_dict['im_id_list']
    ts_ng_path_list = ts_ng_read_dict['im_path_list']
    ts_num_ng = len(ts_ng_id_list)

    # get train/test label
    train_ok_label = np.ones(tr_num_ok)
    train_ng_label = np.zeros(tr_num_ng)
    test_ok_label = np.ones(ts_num_ok)
    test_ng_label = np.zeros(ts_num_ng)
    
    # get train/test list
    tr_ok_img_list = [tr_ok_path_list[j] for j in tr_ok_id_list]
    tr_ng_img_list = [tr_ng_path_list[j] for j in tr_ng_id_list]
    ts_ok_img_list = [ts_ok_path_list[j] for j in ts_ok_id_list]
    ts_ng_img_list = [ts_ng_path_list[j] for j in ts_ng_id_list]
    tr_img_list = tr_ok_img_list + tr_ng_img_list

    # Total training images and annotations
    tr_img_list = tr_ok_img_list + tr_ng_img_list
    tr_label = np.concatenate([train_ok_label, train_ng_label], 0)

    # Total testing images and annotations
    ts_img_list = ts_ok_img_list + ts_ng_img_list
    ts_label = np.concatenate([test_ok_label, test_ng_label], 0) 

    # Initialize model
    model_path = param['model_dir'] + '/dunet_' + param['num_test_iters'] + '.ckpt'
    if_dunet = IF_DUNET(model_path)
    if_dunet.init_model(input_shape = [1, param['load_height'], param['load_width'], 3], gpu_id = '1', per_gpu_fraction = 0.9)
    
    print('Performing inference for training set...')
    # Calculate prediction score of training images.
    tr_output = np.zeros(tr_num_ok + tr_num_ng)
    #tr_output_fea1 = np.zeros([tr_num_ok + tr_num_ng, 2048])
    #tr_output_fea2 = np.zeros([tr_num_ok + tr_num_ng, 2048])
    #for i in range(10):
    for i in range(tr_num_ok + tr_num_ng):
        img = load_data_from_img_list([tr_img_list[i]], param['load_width'], param['load_height'], param['pixel_means']) # 1 * 256 * 256 * 3
        # img = img[np.newaxis, :]
        tr_output_image_tmp, tr_fea1, tr_fea2, tr_output_tmp = if_dunet.inference(if_dunet.graph, if_dunet.sess, img)
        tr_output[i] = tr_output_tmp[0][0]
        #tr_output_fea1[i, :] = tr_fea1
        #tr_output_fea2[i, :] = tr_fea2
    tr_output = [round(val, 6) for val in tr_output] # output is a numpy.ndarray
    #print('tr_output_label is: ', tr_output)
    
    print('Get threshold from max_ok...')
    # Define thresh type and get thresh
    #thresh_type = 'min_ng'
    thresh_type = 'max_ok'
    thresh_val, ind_val = get_thresh(tr_output, tr_label, thresh_type)
    print("threshold is: ", thresh_val)
    thresh_img = load_data_from_img_list([tr_img_list[ind_val]], param['load_width'], param['load_height'], param['pixel_means']) # 1 * 256 * 256 * 3
    thresh_img = (thresh_img[0] * 2 - 1) * 255 + param['pixel_means']  # 256 * 256 * 3
    # print(thresh_img.shape)
    cv2.imwrite('thresh.png', thresh_img)

    print('Performing inference for testing set...')
    # test image from testing set
    output_result_dir = param['output_dir'] + '/result'
    if not os.path.isdir(output_result_dir):
        os.makedirs(output_result_dir) 
    output_img_dir = param['output_dir'] + '/test'
    if not os.path.isdir(output_img_dir):
        os.makedirs(output_img_dir) 

    # Calculate prediction score of testing images.
    ts_output = np.zeros(ts_num_ok + ts_num_ng)
    #ts_output_fea1 = np.zeros([ts_num_ok + ts_num_ng, 2048])
    #ts_output_fea2 = np.zeros([ts_num_ok + ts_num_ng, 2048])
    for i in range(ts_num_ok + ts_num_ng):
        img = load_data_from_img_list([ts_img_list[i]], param['load_width'], param['load_height'], param['pixel_means'])
        # img = img[np.newaxis, :]
        ts_output_image_tmp, ts_fea1, ts_fea2, ts_output_tmp = if_dunet.inference(if_dunet.graph, if_dunet.sess, img)
        tmp_img = (ts_output_image_tmp[0, :, :, :] * 2 - 1) * 255 + param['pixel_means']
        tmp_path = os.path.join(output_img_dir, ts_img_list[i])
        cv2.imwrite(tmp_path, tmp_img)
        ts_output[i] = ts_output_tmp[0][0]
        #ts_output_fea1[i, :] = ts_fea1
        #ts_output_fea2[i, :] = ts_fea2
    ts_output = [round(val, 6) for val in ts_output] # output is a numpy.ndarray
    #print('ts_output_label is: ', ts_output)

    # Evaluate train set
    tr_recall, tr_precision, tr_f1_score = get_recall_precision(tr_output, tr_label, thresh_type, thresh_val)
    print('tr recall is %f tr precision is %f, tr f1_score is %f.'%(tr_recall, tr_precision, tr_f1_score))
    #tr_ok_ng_array = tr_output < thresh_val 

    # Evaluate test set
    ts_recall, ts_precision, ts_f1_score = get_recall_precision(ts_output, ts_label, thresh_type, thresh_val)
    print('ts recall is %f ts precision is %f, ts f1_score is %f.'%(ts_recall, ts_precision, ts_f1_score))
    ts_ok_ng_array = ts_output < thresh_val


    # write features and labels to disk
    np.save(param['output_dir'] + '/result' + '/tr_label.npy', tr_label)
    np.save(param['output_dir'] + '/result' + '/ts_label.npy', ts_label)
    np.save(param['output_dir'] + '/result' + '/tr_score.npy', tr_output)
    np.save(param['output_dir'] + '/result' + '/ts_score.npy', ts_output)

    #np.save(param['output_dir'] + '/result' + '/tr_fea_before_enc.npy', tr_output_fea1)
    # np.save(param['output_dir'] + '/result' + '/ts_fea_before_enc.npy', ts_output_fea1)
    #np.save(param['output_dir'] + '/result' + '/tr_fea_after_enc.npy', tr_output_fea2)
    # np.save(param['output_dir'] + '/result' + '/ts_fea_after_enc.npy', ts_output_fea2)

