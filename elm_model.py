# -*- coding: utf-8 -*-
"""

Last update  February 5, 2019 
Training for the case less samples than neurons implemented 
Multiquadric nonlinearity added 

Note: This ELM is particularly suited for low-complexity 
implementations, with 2 bit quantization on the first layer 
and 8-16 on the output layer 
Also the recommended nonlinearity is 3 (absolute value)
somhow replacing the multiquadric 

First release: June 6, 2018 
@authors: radu.dogaru@upb.ro ioana.dogaru@upb.ro
 
Implements ELM training using datasets available in Matlab format 
Similar to the Octave / Matlab implementation 
Tested under Python 3.6 (Anaconda 5.1)

Software supporting the article: 
[1] Radu Dogaru*, Ioana Dogaru, "Optimized Extreme Learning Machine for Big Data
Applications using Python", in COMM-2018, The 12th International Conference 
on Communications, Bucharest, Romania, 14-16 June 2018. 

Please cite the above article in works where this software is used
"""

import numpy as np
import scipy.io as sio
import scipy.linalg
import time as ti 


def hidden_nonlin(hid_in, tip):
# implementation of the hidden layer 
# additional nonlinearitys may be added 
    if tip==0: 
        # sigmoid 
        H=np.tanh(hid_in)        
    elif tip==1:
        # linsat 
        H=abs(1+hid_in)-abs(1-hid_in)
    elif tip==2:
        # ReLU
        H=abs(hid_in)+hid_in
    elif tip==3:
        # see [1] - very well suited for emmbeded systems 
        H=abs(hid_in)
    elif tip==4:
        H=np.sqrt(hid_in*hid_in+1)
        # multiquadric 
    return H
        

def elmTrain_optim(X, Y, h_Neurons, C , tip):
# Training phase - floating point precision (no quantization)
# X - Samples (feature vectors) Y - Labels

    '''
    Ntr
    Number of train examples,each columns represents a train example so we must
    count the number of elements in a row to know how many train examples we have 
    '''
    Ntr = np.size(X,1) 
    '''
    We have the number of input neurons in the number of rows, so we count the number of elements 
    in a column
    '''
    in_Neurons = np.size(X,0) 
    
    classes = np.max(Y)
    # transforms label into binary columns  
    targets = np.zeros( (classes, Ntr), dtype='int8' )
    for i in range(0,Ntr):
       targets[Y[i]-1, i ] = 1
    targets = targets * 2 - 1
      
      #   Generate inW layer  
    rnd = np.random.RandomState()
    inW=-1+2*rnd.rand(h_Neurons, in_Neurons).astype('float32')
    #inW=rnd.randn(nHiddenNeurons, nInputNeurons).astype('float32')
      
    #  Compute hidden layer 
    hid_inp = np.dot(inW, X)
    H=hidden_nonlin(hid_inp,tip)
      
    # Moore - Penrose computation of output weights (outW) layer 
    if h_Neurons<Ntr:
        print('LLL - Less neurons than training samples')
        outW = scipy.linalg.solve(np.eye(h_Neurons)/C+np.dot(H,H.T), np.dot(H,targets.T))     
    else:
        print('MMM - More neurons than training samples')
        outW = np.dot(H,scipy.linalg.solve(np.eye(Ntr)/C+np.dot(H.T,H), targets.T))
    return inW, outW 

# implements the ELM training procedure with weight quantization       
def elmTrain_fix( X, Y, h_Neurons, C , tip, ni):
# Training phase - emulated fixed point precision (ni bit quantization)
# X - Samples (feature vectors) Y - Labels
# ni - number of bits to quantize the inW weights 
      Ntr = np.size(X,1)
      in_Neurons = np.size(X,0)
      classes = np.max(Y)
      # transforms label into binary columns  
      targets = np.zeros( (classes, Ntr), dtype='int8' )
      for i in range(0,Ntr):
          targets[Y[i]-1, i ] = 1
      targets = targets * 2 - 1
      
      #   Generare inW 
      #   Generate inW layer  
      rnd = np.random.RandomState()
      inW=-1+2*rnd.rand(h_Neurons, in_Neurons).astype('float32')
      #inW=rnd.randn(nHiddenNeurons, nInputNeurons).astype('float32')
      Qi=-1+pow(2,ni-1) 
      inW=np.round(inW*Qi)
      
      #  Compute hidden layer 
      hid_inp = np.dot(inW, X)
      H=hidden_nonlin(hid_inp,tip)
     
      # Moore - Penrose computation of output weights (outW) layer 
      if h_Neurons<Ntr:
          print('LLL - Less neurons than training samples')
          outW = scipy.linalg.solve(np.eye(h_Neurons)/C+np.dot(H,H.T), np.dot(H,targets.T))     
      else:
          print('MMM - More neurons than training samples')
          outW = np.dot(H,scipy.linalg.solve(np.eye(Ntr)/C+np.dot(H.T,H), targets.T))
     
      return inW, outW 
      

def elmPredict_optim( X, inW, outW, tip):
# implements the ELM predictor given the model as arguments 
# model is simply given by inW, outW and tip 
# returns a score matrix (winner class has the maximal score)

      hid_in=np.dot(inW, X)
      H=hidden_nonlin(hid_in,tip)
      score = np.transpose(np.dot(np.transpose(H),outW))
      return score 
        