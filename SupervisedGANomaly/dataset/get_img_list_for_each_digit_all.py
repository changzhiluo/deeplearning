Warning: Do not run me!!!!

import os
import numpy
import json
import random
from get_img_list_for_each_digit import get_img_list

digit_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ratio_list = [10, 100, 1000, 10000]
split_list = [1, 2, 3] # randomly split groups

for target_digit in digit_list:
	print("Processing digit %s..."%(target_digit))
	for ok_ng_ratio in ratio_list:
		for split_num in split_list:
			get_img_list(target_digit, ok_ng_ratio, split_num)
