import numpy as np 
from random import randint 

array = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
labels = np.array([[5,6,7,8]])
print(np.shape(labels))


def mix_columns(array,labels):
    (nr_rows, nr_columns) = np.shape(array)

    for _ in range(nr_columns):
        rand1 = randint(0,nr_columns-1)
        print("rand1: ",rand1)
        rand2 = randint(0,nr_columns-1)
        print("rand2: ",rand2)
        
        tmp_array = np.copy(array[:,rand1])
        tmp_labels = np.copy(labels[:,rand1])
        
       
        array[:,rand1] = array[:,rand2]
        labels[:,rand1] = labels[:,rand2]
        
        array[:,rand2] = tmp_array
        labels[:,rand2] = tmp_labels 
        
    return array,labels
       
   
