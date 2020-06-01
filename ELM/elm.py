import sys
import os
sys.path.append(os.getcwd()+'\..')

from elm_model import *
# ======================================================
#  RUNNING EXAMPLE 
#================================================================================
# parameters 

# parameters 
nume='vr_train_set' # Database (Matlab format - similar to what is supported by the LIBSVM library)
#nume='mnist' # MNIST dataset 
nr_neuroni= 5000 # Proposed number of neurons on the hidden layer 
C=0.100 # Regularization coefficient C  
tip=3 # Nonlinearity of the hidden layer  
nb_in=2  # 0 = float; x - represents weights on a finite x number of bits 
nb_out=0 # same as above but for the output layer

#===============  TRAIN DATASET LOADING ==========================================
# converts into 'float32' for faster execution 
t1 = ti.time()
db=sio.loadmat(nume+'.mat')
Samples=db['Samples'].astype('float32')
Labels=db['Labels'].astype('int8')
clase=np.max(Labels)
trun = ti.time()-t1
print(" load train data time: %f seconds" %trun)

#================= TRAIN ELM =====================================================
t1 = ti.time()
if nb_in>0:
    inW, outW = elmTrain_fix(Samples, np.transpose(Labels), nr_neuroni, C, tip, nb_in)
else:
    inW, outW = elmTrain_optim(Samples, np.transpose(Labels), nr_neuroni, C, tip)
trun = ti.time()-t1
print(" training time: %f seconds" %trun)

# ==============  Quantify the output layer ======================================
Qout=-1+pow(2,nb_out-1)
if nb_out>0:
     O=np.max(np.abs(outW))
     outW=np.round(outW*(1/O)*Qout)


input_output_weights = {

    "inW": inW,
    "outW": outW
}

sio.savemat("io_weights.mat",input_output_weights)

#================= TEST (VALIDATION) DATASET LOADING 
t1 = ti.time()
db=sio.loadmat('vr_test_set.mat')
Samples=db['Samples'].astype('float32')
Labels=db['Labels'].astype('int8')
n=Samples.shape[0]
N=Samples.shape[1]
trun = ti.time()-t1
print( " load test data time: %f seconds" %trun)
#====================== VALIDATION PHASE (+ Accuracy evaluation) =================
t1 = ti.time()
scores = elmPredict_optim(Samples, inW, outW, tip)
trun = ti.time()-t1
print( " prediction time: %f seconds" %trun)
print("scores: ",scores)
command_list = ['avarii', 'claxon', 'frana', 'hey_jarvis', 'inainte', 'inapoi', 'lumini', 'radio', 'start', 'stop']

if __name__ == "__main__":

    #CONFUSION MATRIX computation ==================================
    Conf=np.zeros((clase,clase),dtype='int16')
    
    
    for i in range(N):
        # gasire pozitie clasa prezisa 
        ix=np.nonzero(scores[:,i]==np.max(scores[:,i]))
        
        pred=int(ix[0])
        actual=Labels[0,i]-1
        Conf[actual,pred]+=1
    accuracy=100.0*np.sum(np.diag(Conf))/np.sum(np.sum(Conf))
    print("Confusion matrix is: ")
    print(Conf)
    print("Accuracy is: %f" %accuracy)
    print( "Number of hidden neurons: %d" %nr_neuroni)
    print( "Hidden nonlinearity (0=sigmoid; 1=linsat; 2=Relu; 3 - ABS; 4- multiquadric): %d" %tip)

    #====================================================================================   

    '''
    Running example (on MNIST)  with 2 bits per weights in the input layer 
    Using MKL-NUMPY / CPU: Intel Core-I7 6700HQ (4-cores @ 2.6Ghz)

     load train data time: 1.328532 seconds
     training time: 25.102763 seconds
     load test data time: 0.314851 seconds
     prediction time: 1.308466 seconds
    Confusion matrix is: 
    [[ 970    1    1    0    0    1    3    1    2    1]
     [   0 1126    2    1    1    0    2    0    3    0]
     [   6    0  987   10    3    0    2    8   14    2]
     [   0    0    2  986    0    6    0    6    6    4]
     [   1    0    2    0  961    0    5    2    2    9]
     [   3    0    0    9    1  866    8    2    1    2]
     [   5    2    1    0    4    4  934    0    8    0]
     [   0    9   12    3    2    1    0  986    3   12]
     [   3    0    2    9    2    2    2    5  945    4]
     [   5    5    3    9   11    5    0    6    1  964]]
    Accuracy is: 97.250000
    Number of hidden neurons: 8000
    Hidden nonlinearity (0=sigmoid; 1=linsat; 2=Relu; 3 - ABS; 4- multiquadric): 3
    inW
    Out[119]: 
    array([[ 0.,  1., -1., ..., -0., -0., -0.],
           [-0.,  1., -0., ...,  1.,  1.,  0.],
           [ 0., -1., -0., ...,  0.,  0., -1.],
           ...,
           [-1., -1.,  0., ..., -0., -1., -1.],
           [-1., -1., -0., ..., -0., -0.,  1.],
           [ 0., -0., -1., ..., -1., -1.,  0.]], dtype=float32)
    '''
    