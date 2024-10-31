# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:45:56 2024

@author: ranik
"""
# importing dependencies
import numpy as np, tensorflow as tf
from scipy.io import loadmat
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from scipy.io import loadmat
data = loadmat("D:\\SEMESTER-3\\Unsupervisedlearning\\PROJECT\\umist_cropped.mat")
# Check the type and keys to understand the structure of the data
print(type(data))          # Should be a dictionary
print(data.keys())          # Shows the keys available in the file
# Assuming 'data' is the dictionary you obtained from loading your file
print("Type of data:", type(data))  # Should be a dictionary
print("Available keys in data:", data.keys())  # Shows the keys available in the file

# Print the header information
header_info = data['__header__']
print("Header Information:")
print(header_info)
print(data["facedat"])
# printing the number of images
print(data["facedat"].shape)
data["facedat"][0].shape
# Flatten the array to get a simple list of strings
#labels = [item[0] for item in dirnames[0]]
#print(labels)  # This will print a list like ['1a', '1b', '1c', ...]
# observe and printing the shape of  each image
for i in range(0,len(data["facedat"][0])):
    print(data["facedat"][0][i].shape)
# initialize data and target with empty list
face, target = list(), list()
# seperating data and target
label = 0
for batch in data:
    for sample in batch.T:
        sample = sample.T.reshape(-1)
        face.append(sample)
        target.append(label)
    del sample
    del batch
    label += 1
face, target = np.array(face), np.array(target)
# using StratifiedShuffleSplit for splitting
from sklearn.model_selection import StratifiedShuffleSplit
splitter=StratifiedShuffleSplit(test_size=0.2,random_state=17) 