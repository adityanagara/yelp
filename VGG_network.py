# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 00:29:33 2016

@author: adityanagarajan
"""

#from scipy import ndimage
import pandas as pd
import numpy as np
import os
import sys

from PIL import Image
from theano import tensor as T
import theano
import lasagne
import deep_network
import time
import csv
import cPickle
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
photos_labels = pd.read_csv('../data/train_photos.csv')


def resize_and_crop(img_path, modified_path, size, crop_type='top'):
    """
    Resize and crop an image to fit the specified size.

    args:
        img_path: path for the image to resize.
        modified_path: path to store the modified image.
        size: `(width, height)` tuple.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'middle' or
            'bottom/right' of the image to fit the size.
    raises:
        Exception: if can not open the file in img_path of there is problems
            to save the image.
        ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    #The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], int(round(size[0] * img.size[1] / img.size[0]))),
                Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, round((img.size[1] - size[1]) / 2), img.size[0],
                   round((img.size[1] + size[1]) / 2))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((int(round(size[1] * img.size[0] / img.size[1])), size[1]),
                Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (round((img.size[0] - size[0]) / 2), 0,
                   round((img.size[0] + size[0]) / 2), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else :
        img = img.resize((size[0], size[1]),
                Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
    img.save(modified_path)
    return np.asarray(img)


def load_images(file_path,dest_path):
    
    arr = resize_and_crop(file_path,file_path,(224,224))
    

#images_path = '../data/data/train_photos/'
###images_list = os.listdir(images_path)
##images_list = filter(lambda x: x[-4:] == '.jpg',images_list)
#start_index = int(sys.argv[1])
#end_index = int(sys.argv[2])
#print 'Starting conversion to (224,224) '
#for i in xrange(start_index,end_index):#photos_labels.shape[0]
#    dest_path = str(photos_labels.photo_id[i]) + '_.jpg'
##    print str(photos_labels.photo_id[i]) + '.jpg'
#    load_images(images_path + str(photos_labels.photo_id[i]) + '.jpg',images_path + dest_path)
#    if i % 1000. == 0.0:
#        print 'Images done : %d '%i
def save_file(file_name):
    file_path = 'output/' + file_name
    with open(file_path,'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['epoch_number','training_loss','validation_loss'])

def append_file(file_name,val_1,val_2,val_3):
    file_path = 'output/' + file_name
    with open(file_path,'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([val_1,val_2,val_3])

    

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_epochs = 100,num_points = 1200,compute_flag='cpu'):
    # Arguments passed as string need to be converted to int    
    num_epochs = int(num_epochs)
    num_points = int(num_points)
    # Define name of output files
    results_file_name = 'exp_' + str(num_epochs) + '_' + str(num_points) + '_' + compute_flag + '.csv'
    network_file_name = 'network_' + str(num_epochs) + '_' + str(num_points) + '_' + compute_flag 
    print 'Saving file to: %s' % results_file_name
    print 'Number of points: %d ' % num_points
    print 'Compute Flag: %s ' % compute_flag
    save_file(results_file_name)  
    Deep_learner = deep_network.DCNN()
    # Define the input tensor
    input_var = T.tensor4('inputs')
    # Define the output tensor (in this case it is a real value or reflectivity)
    output_var = T.ivector('targets')
    # User input to decide which experiment to run, cpu runs were performed
    # to check if the network was working correctly
    network = Deep_learner.build_DCNN(input_var)
    
    train_prediction = lasagne.layers.get_output(network)
    test_prediction = lasagne.layers.get_output(network)
   
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, output_var)
    
    loss = loss.mean()
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)
    
    train_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), output_var),
                      dtype=theano.config.floatX) 
    # Define theano function which generates and compiles C code for the optimization problem
    train_fn = theano.function([input_var, output_var], [loss,train_acc], updates=updates)
    
#    test_fn = theano.function([input_var, output_var],test_loss, updates=updates)
    
    base_path = '/home/an67a/deep_nowcaster/data/dataset2/'
    training_set_list = os.listdir(base_path)
    training_set_list = filter(lambda x: x[-4:] == '.pkl' and 'val' not in x,training_set_list)
    validation_set_list = os.listdir(base_path)
    validation_set_list = filter(lambda x: x[-4:] == '.pkl' and 'val' in x,validation_set_list)
    experiment_start_time = time.time()
    # Load Data Set
    DataSet = []
    print('Loading data set...')
    for file_name in training_set_list[:3]:
        print file_name
        temp_file = file(base_path + file_name,'rb')
        X_train,Y_train = cPickle.load(temp_file)
        temp_file.close()
        Y_train = Y_train.reshape(-1,).astype('uint8')
        DataSet.append((X_train,Y_train))
    
    print('Start training...')
    for epoch in range(num_epochs):
        print('Epoch number : %d '%epoch)
        train_err = 0
        train_batches = 0
        train_acc = 0
        start_time = time.time()
        for data in DataSet:
#        for file_name in training_set_list:
#            print file_name
#            temp_file = file(base_path + file_name,'rb')
#            X_train,Y_train = cPickle.load(temp_file)
#            Y_train = Y_train.astype('uint8')
#            temp_file.close()
            for batch in iterate_minibatches(data[0], data[1], 1059, shuffle=False):
                inputs, targets = batch
                err,acc = train_fn(inputs, targets)
                train_err += err
                train_acc += acc
                train_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        append_file(results_file_name,epoch + 1,round(train_err / train_batches,2),round((train_acc / train_batches) * 100,2))
        
        # Dump the network file every 100 epochs
        if (epoch + 1) % 100 == 0:
            print('creating network file')
            network_file = file('/home/an67a/deep_nowcaster/output/'+ network_file_name + '_' + str(epoch + 1) + '.pkl','wb')
            cPickle.dump(network,network_file,protocol = cPickle.HIGHEST_PROTOCOL)
            network_file.close()
    time_taken = round(time.time() - experiment_start_time,2)
    print('The experiment took {:.3f}s'.format(time.time() - experiment_start_time))
    append_file(results_file_name,'The experiment took',time_taken,0)


if __name__ == '__main__':
    kwargs['experiment'] = sys.argv[1]
    main(**kwargs)
    
    print 'Done!' 

    
'''
234842
'''



