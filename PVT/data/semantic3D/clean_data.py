import numpy as np
import time
import math
import os

#raw_folder = 'C:\\Users\\Andrew\\Desktop\\CS674\\FinalProject\\full_dataset\\train\\raw\\'
#clean_folder = 'C:\\Users\\Andrew\\Desktop\\CS674\\FinalProject\\full_dataset\\train\\clean\\'
base_folder = os.path.dirname(os.path.abspath(__file__))
print(base_folder)
raw_folder = os.path.join(base_folder, 'full_dataset\\train\\raw\\')
clean_folder = os.path.join(base_folder, 'full_dataset\\train\\clean\\')
def clean_data(data_name, training=True):
    start_time = time.time()
    print(f'{time.ctime(start_time)}: Loading datafile {data_name}')
    data = np.loadtxt(raw_folder + 'points\\' + data_name + '.txt')
    print(f'{time.ctime(time.time())}: Loading labels for {data_name}')
    labels = np.loadtxt(raw_folder + 'labels\\' + data_name + '.labels')
    
    undef_mask = [labels[i]!=0 for i in range(len(labels))]

    data = data[undef_mask]
    labels = labels[undef_mask]
    # our label started at 1, this subtraction changes it to start at 0
    labels = [x - 1 for x in labels]

    end_time = time.time()
    print(f'{time.ctime(end_time)}: Saving cleaned files! Done in {end_time-start_time} seconds!')
    np.save(clean_folder + 'points\\' + data_name + '.npy', data)
    np.save(clean_folder + 'labels\\' + data_name + '.npy', labels)

raw_files_path = raw_folder+'points'
clean_files_path = clean_folder+'points'
print(raw_files_path)
for f in os.listdir(raw_files_path):
    print(f)
raw_files = [f[:-4] for f in os.listdir(raw_files_path) if os.path.isfile(os.path.join(raw_files_path, f))]
print(raw_files)
for file in raw_files:
    if os.path.exists(os.path.join(clean_files_path,file)+'.npy'):
        print(f'{file} already cleaned! Skipping ...')
        continue
    clean_data(file)