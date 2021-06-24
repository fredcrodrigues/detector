import os
import glob
from posix import listdir

cwd = os.getcwd()
path = cwd + '/data/'


list_train = []
list_test = []

for name_dir in os.listdir(path):
    
    a_train = open('data/train.txt' , 'a')
    a_test = open('data/val.txt' , 'a')

    if name_dir == 'train':
        path_dir = path  + name_dir
        #print(path_dir)
        for name_train in sorted(glob.glob(path_dir + '/*.JPG')):
            list_train.append(name_train + '\n')

    if name_dir == 'val':
        path_dir = path  + name_dir
        for name_val in sorted(glob.glob(path_dir + '/*.JPG')):
            list_test.append(name_val + '\n')


    a_train.writelines(list_train)
    a_test.writelines(list_test)

    a_train.close()
    a_test.close()
        
