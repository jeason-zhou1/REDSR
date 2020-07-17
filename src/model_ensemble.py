import cv2
import os 
import numpy as np

save_path = '../AIM_ValidationX4'
# save_path = '../train_SR/result_ED_S'
root_path = '../train_SR/EDSR/experiment/'
path = 'results-AIM'

os.makedirs(save_path,exist_ok=True)

paths = [root_path+'AIM_TEST_1/'+path,root_path+'AIM_TEST_2/'+path,root_path+'AIM_TEST_3/'+path,root_path+'AIM_TEST_4/'+path]
paths = ['../train_SR/EDSR/experiment/AIM_TEST_3/results-AIM','../train_SR/result_2','../train_SR/EDSR/experiment/AIM_TEST_X4/results-AIM']
paths = ['../train_SR/EDSR/experiment/AIM_TEST_X4/results-AIM','../train_SR/result_ED']
paths = ['../experiment/AIM_EDSR_FINAL_X4/results-Demo','../experiment/AIM_DRLN_FINAL_X4/results-Demo']

nums = len(paths)

list_pic = [[] for _ in range(nums)]
files = os.listdir(paths[0])
for i, each in enumerate(files):
    for path_num, j in enumerate(paths):
        each_pic = cv2.imread(j+'/'+each)
        # print(i,each,type(each_pic))
        list_pic[path_num] = each_pic
    
    array = np.asarray(list_pic)
    y = array.mean(axis=0)
    print(y.shape,save_path+'/'+each[:-8]+'.png')
    cv2.imwrite(save_path+'/'+each[:-8]+'.png',y)
