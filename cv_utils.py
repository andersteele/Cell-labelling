import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def describe(X,Y,Y_pred, label = None):
    Y_pred_im = Y_pred.argmax(3)
    Y_im = Y.argmax(3)
    
    plt.figure(1)
    plt.title('Input')
    plt.imshow(X[0,:,:,0])
    #plt.colorbar()

    plt.figure(2)
    plt.title('Ground Truth')
    plt.imshow(Y_im[0,:,:])
    #plt.colorbar()

    plt.figure(3)
    plt.title('Predicted')
    plt.imshow(Y_pred_im[0,:,:])
    
    if label:
        plt.figure(4)
        plt.title('Label probability')
        plt.imshow(Y_pred[0,:,:,label])
    #plt.colorbar()
    print('Accuracy is')
    print(np.sum(Y_im[0,:,:] == Y_pred_im[0,:,:])/(256*256.))
    
def cross_validate(model, cv_gen, no_samples = 100, outlier_cut = .95):
    accuracy_list = []
#cv_list = zip([cell_3],[simple_label_3])
    c =1
    slice_list = []
    outliers = []
    for _ in xrange(no_samples):
        X_t,Y_t = next(cv_gen)
        Y_pred = model.predict(X_t)
        Y_pred_im = Y_pred.argmax(3)
        Y_im = Y_t.argmax(3)
        accuracy = np.sum(Y_im == Y_pred_im)/(256*256.)
        accuracy_list.append(accuracy)
        slice_list.append((X_t,Y_t,Y_pred))
        if accuracy < outlier_cut:
            describe(X_t,Y_t,Y_pred)

    acc = pd.DataFrame(accuracy_list)
    return(acc, outliers)
    
def sample(model, gen, label = None):
    X,Y = next(gen)
    Y_pred = model.predict(X)
    describe(X,Y,Y_pred, label)
    return X,Y,Y_pred
