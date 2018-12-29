#---------------------------------------------
# Function: Define commonly used data io operations
# Author:   Changzhi Luo
# Date:     20180908
#---------------------------------------------
import os
import cv2
import numpy as np
import random

# Load data from a specified directory and conduct normalization operation.
def load_data(img_dir, image_width, image_height, pixel_means):
    if not os.path.isdir(img_dir):
        print('Image folder does not exist.', img_dir)
        return None
    image_name_list = os.listdir(img_dir)
    image_num = len(image_name_list)
    data_mat = np.zeros([image_num, image_height, image_width, 3]) # notice the width/height order
    for ind, image_name in enumerate(image_name_list):
        im = cv2.imread(os.path.join(img_dir, image_name))
        im = im.astype(np.float, copy=False)
        im -= pixel_means 
        im = ((im / 255.0 ) + 1) / 2 # normalize to [0, 1]
        im2 = cv2.resize(im, (image_width, image_height)) # # notice the width/height order
        data_mat[ind] = im2 # np.array
    return data_mat

# Load data from a specified image list and conduct normalization operation.
def load_data_from_img_list(img_list, image_width, image_height, pixel_means):
    # img_list consists of the full path of each image
    assert(type(img_list) == list and len(img_list) != 0)
    data_mat = np.zeros([len(img_list), image_height, image_width, 3]) # notice the width/height order
    for ind in range(len(img_list)):
        im = cv2.imread(img_list[ind])
        im = im.astype(np.float, copy=False)
        im -= pixel_means 
        im = ((im / 255.0 ) + 1) / 2 # normalize to [0, 1]
        im2 = cv2.resize(im, (image_width, image_height)) # # notice the width/height order
        data_mat[ind] = im2 # np.array
    return data_mat

# Write resultant images into a specified directory.
def write_image(write_dir, ori_img_data, reconstruct_img_data, ok_ng_array, score_array): # write image ndarray (N, h, w, c) into img_dir
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)   
    reconstruct_error_img = ori_img_data - reconstruct_img_data
    for i in range(reconstruct_img_data.shape[0]):
        cv2.putText(reconstruct_img_data[i],str(score_array[i]),(0,50),cv2.FONT_HERSHEY_SIMPLEX, 2 ,(255,0,0),2) # 分数打到图片上
        # 画框， ng红，ok绿
        if(ok_ng_array[i] == False): # ng
            cv2.rectangle(reconstruct_img_data[i], (0, 0), (reconstruct_img_data[i].shape[1], reconstruct_img_data[i].shape[0]), (0, 0, 255), 5) # bgr
        else:
            cv2.rectangle(reconstruct_img_data[i], (0, 0), (reconstruct_img_data[i].shape[1], reconstruct_img_data[i].shape[0]), (0, 255, 0), 5) # bgr
    
    reconstruct_error_img = (reconstruct_error_img + 1) / 2 # normalize to [0, 1], otherwise has negative numbers
    concat_img = np.concatenate([ori_img_data, reconstruct_img_data, reconstruct_error_img], 2)
    concat_img *= 255.0
    for i in range(concat_img.shape[0]):
        file_path = write_dir + "/" + str(i) + ".png"
        cv2.imwrite(file_path, concat_img[i])

# get mask from rle code (start from 0) in json
def get_mask_from_rle(bbox, rle_lst):
    [x, y, w, h] = [bbox[0], bbox[1], bbox[2], bbox[3]]
    data_lst = [] # save the decoded list
    for rle_ind in range((len(rle_lst))):
        for pix_ind in range(rle_lst[rle_ind]):
            # data_lst.append(rle_ind % 2) # 0,1看不出差别
            data_lst.append((rle_ind % 2) * 255)
    assert(len(data_lst) == (w * h)) # ensure the decoding process is correct
    data_array = np.array(data_lst, dtype = np.uint8).reshape(h, w)
    return data_array
    

# flip augmentation
def random_flip(data_mat, hflip = 1, vflip = 1): # data_mat is a 4-d ndarray, e.g. (100, 256, 448, 3)
    data_out = np.zeros([data_mat.shape[0], data_mat.shape[1], data_mat.shape[2], data_mat.shape[3]])
    for ind in range(data_mat.shape[0]):
        im = data_mat[ind, :, :, :] # 3-d ndarray (256, 448, 3)
        if hflip == 1 and vflip == 1:
            hrand = random.randint(0, 1) # 随机生成0或1
            vrand = random.randint(0, 1) # 随机生成0或1
            if hrand == 1:
                im = im[:, ::-1, :]
            if vrand == 1:
                im = im[::-1, :, :]
        elif hflip == 1 and vflip == 0:
            hrand = random.randint(0, 1) # 随机生成0或1
            if hrand == 1:
                im = im[:, ::-1, :]
        elif hflip == 0 and vflip == 1:
            vrand = random.randint(0, 1) # 随机生成0或1
            if vrand == 1:
                im = im[::-1, :, :]
        data_out[ind] = im
    return data_out

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)