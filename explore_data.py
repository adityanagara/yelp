# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:36:40 2016

@author: adityanagarajan
"""

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

train_photos_biz_ids = pd.read_csv('../data/train_photo_to_biz_ids.csv')
train_labels = pd.read_csv('../data/train.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')

train_labels.rename(columns={'labels': 'labels_'}, inplace=True)
train_labels = train_labels.dropna(subset = ['labels_'])

arr = train_labels.values

#train_labels = train_labels.reindex(np.arange(train_labels.shape[0]))
#im = Image.open(os.path.join('../input/','train_photos',''.join([str(train_photos.photo_id[0]),'.jpg'])))

#plt.imshow(im)

labels_ = arr[:,1]
labels = np.zeros((arr.shape[0],9),dtype='int16')

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

for x in xrange(labels.shape[0]):
    labels[x,[int(i) for i in arr[x,1].split()]] = 1

y = np.array([bool2int(x[::-1]) for x in labels])
y_encode = np.zeros((172,2),dtype = 'int16')
y_encode[:,0] = np.arange(172)
y_encode[:,1] = np.unique(y)
arr = np.concatenate((arr,y.reshape(y.shape[0],1)),axis=1)
temp_labels = np.zeros(arr.shape[0],dtype='int16')

for x in xrange(arr.shape[0]):
    temp_labels[x] = y_encode[y_encode[:,1] == arr[x,2]][0][0]

arr = np.concatenate((arr,temp_labels.reshape(temp_labels.shape[0],1)),axis=1)

print 'labeling images...'
#for x in xrange(train_photos_biz_ids.shape[0]):
#    # Find the class label from the array to associate it with an image
#    if arr[train_photos_biz_ids.business_id[x] == arr[:,0],3]:
#        train_photos_biz_ids.loc[x,'class_labels'] = arr[train_photos_biz_ids.business_id[x] == arr[:,0],3]
#
## Save the class labels of the images
#train_photos_biz_ids.to_csv('../data/train_photos.csv')
#print train_photos_biz_ids.shape


#
#print train_labels.shape


    

#y_encode = np.linspace(np.min(np.unique(y)),np.max(np.unique(y)),172)
#y = np.argmin(np.abs(y.reshape(1996,1) - y_encode.reshape(1,172)),axis = 1)
#
#train_labels.loc[:,'labels_'] = y



# encode 






