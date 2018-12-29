import os
import numpy
import json
import random
def get_num(target_digit, ok_ng_ratio, split_num):

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
    im_list = list(filter(lambda x: (('_' + target_digit) in x) == False, im_list_all))
    random_ng_sample_num = int(len(im_list)/ok_ng_ratio) # ng sample num should calculated here using tr_ok_num
    im_id_list = []
    for ind, im_name in enumerate(im_list):
        im_id_list.append(ind)
    print('tr_ok num is %d.'%len(im_id_list))

    # tr_ng, randomly get random_ng_sample_num ng samples
    im_dir = '/home/luochangzhi/code/supervised_ganomaly/dataset/' + data_dir + '/train'
    im_list_all = os.listdir(im_dir)
    im_list = list(filter(lambda x: (('_' + target_digit) in x) == True, im_list_all))
    im_id_list = []
    for ind, im_name in enumerate(im_list[0:random_ng_sample_num]):
        im_id_list.append(ind)
    print('tr_ng num is %d.'%len(im_id_list))

    # ts_ok
    im_dir = '/home/luochangzhi/code/supervised_ganomaly/dataset/' + data_dir + '/test'
    im_list_all = os.listdir(im_dir)
    im_list = list(filter(lambda x: (('_' + target_digit) in x) == False, im_list_all))
    im_id_list = []
    for ind, im_name in enumerate(im_list):
        im_id_list.append(ind)
    print('tr_ok num is %d.'%len(im_id_list))

    # tr_ng
    im_dir = '/home/luochangzhi/code/supervised_ganomaly/dataset/' + data_dir + '/test'
    im_list_all = os.listdir(im_dir)
    im_list = list(filter(lambda x: (('_' + target_digit) in x) == True, im_list_all))
    im_id_list = []
    for ind, im_name in enumerate(im_list):
        im_id_list.append(ind)
    print('ts_ng num is %d.'%len(im_id_list))
