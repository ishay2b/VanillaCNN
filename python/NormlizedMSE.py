# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:57:25 2015

@author:Ishay Tubi
"""

import caffe
import numpy as np


class NormlizedMSE(caffe.Layer):
    """
    Compute the normlized MSE Loss 
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
 #       self.diff = np.zeros(bottom[0].count,dtype='float32')
        # loss output is scalar
        top[0].reshape(1)
      #  print('NormlizedMSE bottom shape ',bottom[0].shape[0],bottom[0].shape[1])
                
    def forward(self,bottom, top):
        '''Mean square error of landmark regression normalized w.r.t.                                                
        inter-ocular distance                                                                                        
        '''
         # Lets assume batch size is 16 in this example remarks
        # input size is (16,10) 
        y_true = bottom[1].data # Assume second entry is the labled data
        y_pred = bottom[0].data
        
        #eye_indices = left eye x, left eye y, right eye x, right eye y, nose x, left mouth, right mouth
        delX = y_true[:,0]-y_true[:,2] # del X size 16
        delY = y_true[:,1]-y_true[:,3] # del y size 16
        self.interOc = (1e-6+(delX*delX + delY*delY)**0.5).T # Euclidain distance
        #Cannot multiply shape (16,10) by (16,1) so we transpose to (10,16) and (1,16) 
        diff = (y_pred-y_true).T # Transpose so we can divide a (16,10) array by (16,1)
        
        self.diff[...]  = (diff/self.interOc).T # We transpose back to (16,10)
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2. # Loss is scalar

    def backward(self, top, propagate_down, bottom):
        
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
#            print(bottom[i].diff[...])

##################################

class EuclideanLossLayer(caffe.Layer):
    #ORIGINAL EXAMPLE
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
            
            
            
if __name__ =="__main__":
    net=caffe.Net('/Users/ishay/Dev/VanilaCNN/train_val_vanila.prototxt', '/Users/ishay/Dev/VanilaCNN/vanilaCNN.caffemodel',caffe.TRAIN)
    prediction = net.forward()['loss'][0]
    print 'lose ', prediction
     