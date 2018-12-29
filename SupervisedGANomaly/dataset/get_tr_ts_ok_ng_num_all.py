import os
import numpy
import json
import random
from get_tr_ts_ok_ng_num import get_num

digit_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ratio_list = [10, 100, 1000, 10000]
split_list = [1] # randomly split groups

for target_digit in digit_list:
	print("Processing digit %s..."%(target_digit))
	for ok_ng_ratio in ratio_list:
		for split_num in split_list:
			get_num(target_digit, ok_ng_ratio, split_num)
