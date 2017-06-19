import numpy as np
from random import randint
from keras.utils import to_categorical

def slice_gen(cell_list, batch_size = 1, buf = 20,axes=[0,2], no_labels = 3, skip = 4, wait = False):
    #cell_list is a list of tuples (stack, labels)
    #stack is the usual cell stack, with noised background
    #labels are 0,1 for not nucleus, nucleus
    #bounding boxes is a list of 3 tuples, (min_i, max_i), where min_i,max_i are min/max index of non-zero element
    #pick a random tuple
    #pick a random axis
    #pick a random depth along the axis that is within bounding box and buffer
    #pick a random 256x256 slice along that depth, contained in bounding box
        #that is, pick two random ints x,y within bounding box -256
        #return that slice
    M = len(cell_list)
    X_out = np.zeros((batch_size,256,256,1))
    Y_out = np.zeros((batch_size,256,256,no_labels))
    while 1:
        for b in range(batch_size):
            cell, labels = cell_list[randint(0,M-1)]
            axis = axes[0]
            if len(axes)>=2:
                #axis = randint(axes[0],axes[1]) #this is wrong
                a = randint(0,len(axes)-1)
                axis = axes[a]
            
            cell = cell.swapaxes(0,axis)
            labels = labels.swapaxes(0,axis)
            assert cell.shape == labels.shape
            depth = randint(buf,cell.shape[0]-buf)
            x0 = skip*randint(0,int((cell.shape[1]-260)/skip))
            x1 = x0 + 256
            y0 = skip*randint(0,int((cell.shape[2]-260)/skip))
            y1 = y0 + 256
            r = randint(0,3)
            X_out[b,:,:,0] = np.rot90(cell[depth,x0:x1,y0:y1],r)
                # we need to convert labels to categorical
                #keras.utils.to_categorical
            Y_out[b,:,:,:] = to_categorical(np.rot90(labels[depth,x0:x1,y0:y1],r),no_labels).reshape((256,256,no_labels))
        yield (X_out,Y_out)
        
def non_zero_gen(cell_list, batch_size = 1, buf = 20,axes=[0,2], no_labels = 3, skip = 4, which_label=1):
    while 1:
        for X,Y in slice_gen(cell_list,batch_size,buf,axes,no_labels,skip):
            if np.sum(Y[0,:,:,which_label])!=0:
                X_out = X
                Y_out = Y
                break
        yield (X,Y)
def non_zero_gen2(**kwargs):
    while 1:
        for X,Y in slice_gen(cell_list,batch_size,buf,axes,no_labels,skip):
            if np.sum(Y[0,:,:,which_label])!=0:
                X_out = X
                Y_out = Y
                break
        yield (X,Y)
