# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 00:10:43 2020

@author: jitup
"""
import shutil
import numpy as np
import os
from os import listdir
from numpy import asarray
from os.path import isdir
import glob

def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

def split(path_to_data, path_to_test_data, train_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))
    
    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - train_ratio))
    
    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save_test = os.path.join(path_to_test_data, 'test',dirs[i])
        
        path_to_save_train = os.path.join(path_to_test_data, 'train',dirs[i])
    
        #creates dir
        if not os.path.exists(path_to_save_test):
            os.makedirs(path_to_save_test)
        if not os.path.exists(path_to_save_train):
            os.makedirs(path_to_save_train)
            
        test_files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save_test, test_files[j])
            src = os.path.join(path_to_original, test_files[j])
            shutil.move(src, dst)
        
        train_files = get_files_from_folder(path_to_original)
        for k in train_files:
            dst_train = os.path.join(path_to_save_train, k)
            src_train = os.path.join(path_to_original, k)
            shutil.move(src_train, dst_train)

    
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = os.path.join(directory,subdir)
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

def load_faces(path):
    faces = list()
	# enumerate files
    for file_path in glob.glob(path + '/*.npy'):
            face = np.load(file_path)
    		# store
            faces.append(face)
    return faces

def load_embeddings_img(directory):
    print("[INFO] loading embeddings...")
    data = []
    for subdir in listdir(directory):
        path = os.path.join(directory,subdir)
        for file_path in glob.glob(path + '/*.npy'):
            enc = np.load(file_path)
            imagePath = file_path.split('.')[0] + '.png'
            
            d = [{"imagePath": imagePath, "embedding": enc}]
            data.extend(d)
    return data

#Split Data into train and test            
# path_to_data=r'C:\Users\jitup\Desktop\Nirovision\faces'
# path_to_test_data=r'C:\Users\jitup\Desktop\Nirovision\Data'
# train_ratio=0.8

# split(path_to_data, path_to_test_data, train_ratio)

