import os
import numpy
import json
import random
def get_img_list(target_digit, ok_ng_ratio, split_num):

    # note that the list path is not start from 0 and end at len
    #target_digit = '9'
    #ok_ng_ratio = 10 #[10, 100, 1000, 10000]
    #split_num = 1
    data_dir = 'data_mnist'
    data_file_dir = 'data_mnist/data_file/' + target_digit + '/ratio_' + \
                    str(ok_ng_ratio) + '/split_' + str(split_num)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)

    # tr_ok
    im_dir = '/home/luochangzhi/code/supervised_ganomaly/dataset/' + data_dir + '/train'
    im_list_all = os.listdir(im_dir)
    # to find all the digits except the target_digit
    im_list = list(filter(lambda x: (('_' + target_digit) in x) == False, im_list_all))
    random_ng_sample_num = int(len(im_list)/ok_ng_ratio) # ng sample num should calculated here using tr_ok_num
    im_id_list = []
    im_path_list = [] 
    for ind, im_name in enumerate(im_list):
        im_id_list.append(ind)
        im_path = os.path.join(im_dir, im_name)
        im_path_list.append(im_path)
    write_dict = dict()
    write_dict['im_id_list'] = im_id_list
    write_dict['im_path_list'] = im_path_list
    with open(data_file_dir + '/tr_ok_path.json', 'w') as f:
        json.dump(write_dict, f)
    #print('Writing %d images into tr_ok set.'%len(im_list))

    # tr_ng, randomly get random_ng_sample_num ng samples
    im_dir = '/home/luochangzhi/code/supervised_ganomaly/dataset/' + data_dir + '/train'
    im_list_all = os.listdir(im_dir)
    # to find the target_digit
    im_list = list(filter(lambda x: (('_' + target_digit) in x) == True, im_list_all))
    random.shuffle(im_list) # shuffle should performed here to allow different splits
    im_id_list = []
    im_path_list = []
    print("ng num is %d."%random_ng_sample_num)
    for ind, im_name in enumerate(im_list[0:random_ng_sample_num]):
        im_id_list.append(ind)
        im_path = os.path.join(im_dir, im_name)
        im_path_list.append(im_path)
    write_dict = dict()
    write_dict['im_id_list'] = im_id_list
    write_dict['im_path_list'] = im_path_list
    with open(data_file_dir + '/tr_ng_path.json', 'w') as f:
        json.dump(write_dict, f)
    #print('Writing %d images into tr_ng set.'%len(im_list))

    # ts_ok
    im_dir = '/home/luochangzhi/code/supervised_ganomaly/dataset/' + data_dir + '/test'
    im_list_all = os.listdir(im_dir)
    im_list = list(filter(lambda x: (('_' + target_digit) in x) == False, im_list_all))
    im_id_list = []
    im_path_list = [] 
    for ind, im_name in enumerate(im_list):
        im_id_list.append(ind)
        im_path = os.path.join(im_dir, im_name)
        im_path_list.append(im_path)
    write_dict = dict()
    write_dict['im_id_list'] = im_id_list
    write_dict['im_path_list'] = im_path_list
    with open(data_file_dir + '/ts_ok_path.json', 'w') as f:
        json.dump(write_dict, f)
    #print('Writing %d images into ts_ok set.'%len(im_list))

    # tr_ng
    im_dir = '/home/luochangzhi/code/supervised_ganomaly/dataset/' + data_dir + '/test'
    im_list_all = os.listdir(im_dir)
    im_list = list(filter(lambda x: (('_' + target_digit) in x) == True, im_list_all))
    im_id_list = []
    im_path_list = [] 
    for ind, im_name in enumerate(im_list):
        im_id_list.append(ind)
        im_path = os.path.join(im_dir, im_name)
        im_path_list.append(im_path)
    write_dict = dict()
    write_dict['im_id_list'] = im_id_list
    write_dict['im_path_list'] = im_path_list
    with open(data_file_dir + '/ts_ng_path.json', 'w') as f:
        json.dump(write_dict, f)
    #print('Writing %d images into ts_ng set.'%len(im_list))
