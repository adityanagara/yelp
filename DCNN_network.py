# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:08:37 2016

@author: adityanagarajan
"""

import lasagne

import numpy as np
import os
import theano
from theano import tensor as T

class DCNN_network:
    
    def __init__(self,input_shape = (None, 3, 224, 224)):
        self.input_shape = input_shape
    
    def build_DCNN(self,input_var = None):
    
        from lasagne.layers import dnn
        print 'We hit the GPU code!'
        # Input image
        l_in = lasagne.layers.InputLayer(shape=self.input_shape,
                                        input_var=input_var)
        # Conv layer 1
        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        # Conv layer 2
        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 0,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        # Max pool 1
        l_maxpool1 = dnn.MaxPool2DDNNLayer(
                        incoming = l_conv2,
                        pool_size = (2,2),
                        stride = (2,2)
                        )
        # Conv layer 3
        l_conv3 = dnn.Conv2DDNNLayer(
            l_maxpool1,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        # Conv layer 4
        l_conv4 = dnn.Conv2DDNNLayer(
            l_conv3,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        # Max pool 2
        l_maxpool2 = dnn.MaxPool2DDNNLayer(
                        incoming = l_conv4,
                        pool_size = (2,2),
                        stride = (2,2)
                        )
        
        # Conv layer 5
        l_conv5 = dnn.Conv2DDNNLayer(
            l_maxpool2,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 6
        l_conv6 = dnn.Conv2DDNNLayer(
            l_conv5,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 7
        l_conv7 = dnn.Conv2DDNNLayer(
            l_conv6,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 8
        l_conv8 = dnn.Conv2DDNNLayer(
            l_conv7,
            num_filters=256,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Max pool 3
        l_maxpool3 = dnn.MaxPool2DDNNLayer(
                        incoming = l_conv8,
                        pool_size = (2,2),
                        stride = (2,2)
                        )
        
        # Conv layer 9
        l_conv9 = dnn.Conv2DDNNLayer(
            l_maxpool3,
            num_filters=512,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 10
        l_conv10 = dnn.Conv2DDNNLayer(
            l_conv9,
            num_filters=512,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 11
        l_conv11 = dnn.Conv2DDNNLayer(
            l_conv10,
            num_filters=512,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 12
        l_conv12 = dnn.Conv2DDNNLayer(
            l_conv11,
            num_filters=512,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        l_maxpool4 = dnn.MaxPool2DDNNLayer(
                        incoming = l_conv12,
                        pool_size = (2,2),
                        stride = (2,2)
                        )
        
        # Conv layer 13
        l_conv13 = dnn.Conv2DDNNLayer(
            l_maxpool4,
            num_filters=512,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 14
        l_conv14 = dnn.Conv2DDNNLayer(
            l_conv13,
            num_filters=512,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 15
        l_conv15 = dnn.Conv2DDNNLayer(
            l_conv14,
            num_filters=512,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        # Conv layer 16
        l_conv16 = dnn.Conv2DDNNLayer(
            l_conv15,
            num_filters=512,
            filter_size=(3, 3),
            stride=(1, 1),
            pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        l_maxpool5 = dnn.MaxPool2DDNNLayer(
                        incoming = l_conv16,
                        pool_size = (2,2),
                        stride = (2,2)
                        )
        
        l_hidden1 = lasagne.layers.DenseLayer(
            l_maxpool5,
            num_units=4096,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        l_dropout1 = lasagne.layers.DropoutLayer(
                        incoming = l_hidden1,
                        p = 0.5
        )
        
        l_hidden2 = lasagne.layers.DenseLayer(
            l_dropout1,
            num_units=4096,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        l_dropout2 = lasagne.layers.DropoutLayer(
                        incoming = l_hidden2,
                        p = 0.5
        )
        
        l_out = lasagne.layers.DenseLayer(
            l_dropout2,
            num_units=172,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        return l_out
    

    
    
    
        
    