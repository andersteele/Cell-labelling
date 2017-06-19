from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Convolution2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPool2D
from keras import metrics
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model
from keras.applications import VGG16
from keras.initializers import Ones

def Skip_Cell_Model(no_labels = 2, opt='adadelta', pass_levels = [1,2,3,4]):
    """
    A deep convolutional neural network for segmenting Xray tomography data
    of cells. The architecture is inspired by U-net,  arxiv:1505.04597.
    We use the VGG-16 architecture and weights for the convolutional layers,
    then upsample by doubling. 

    Returns:
    A keras model, pre-compiled
    
    Keyword arguments:
    no_labels --  number of labelled classes
    opt -- otpimizer to be used for training
    pass_levels -- the skip through layers to include. Lower layers
            include higher resolution data
    """

    bottle_neck = VGG16(include_top=False, weights='imagenet')
    img_input = Input(shape=(256,256,1))
#The initial VGG 16 layers, with skip throughs added
    x = Conv2D(3,(1,1),padding = 'same',input_shape=(256,256,1),
            use_bias=False, kernel_initializer=Ones())(img_input)

    x = Conv2D(**(bottle_neck.layers[1].get_config()))(x)
    x_split_1 = Conv2D(**(bottle_neck.layers[2].get_config()))(x)
    x = MaxPool2D(**(bottle_neck.layers[3].get_config()))(x_split_1)

    x = Conv2D(**(bottle_neck.layers[4].get_config()))(x)
    x_split_2 = Conv2D(**(bottle_neck.layers[5].get_config()))(x)
    x = MaxPool2D(**(bottle_neck.layers[6].get_config()))(x_split_2)

    x = Conv2D(**(bottle_neck.layers[7].get_config()))(x)
    x = Conv2D(**(bottle_neck.layers[8].get_config()))(x)
    x_split_3 = Conv2D(**(bottle_neck.layers[9].get_config()))(x)
    x = MaxPool2D(**(bottle_neck.layers[10].get_config()))(x_split_3)

    x = Conv2D(**(bottle_neck.layers[11].get_config()))(x)
    x = Conv2D(**(bottle_neck.layers[12].get_config()))(x)
    x_split_4 = Conv2D(**(bottle_neck.layers[13].get_config()))(x)
    x = MaxPool2D(**(bottle_neck.layers[14].get_config()))(x_split_4)



    x = Convolution2DTranspose(512,3,padding='same',activation = 'relu')(x)
    x = Convolution2DTranspose(512,3,padding='same',activation = 'relu')(x)
    x = Convolution2DTranspose(512,3,padding='same',activation = 'relu')(x)
    x_join_4 = UpSampling2D((2,2))(x)

    if 4 in pass_levels:
        x = concatenate([x_join_4,x_split_4],axis=3)
    else:
        x = x_join_4
    x = Convolution2DTranspose(512,3,padding='same',activation = 'relu')(x)
    x = Convolution2DTranspose(512,3,padding='same',activation = 'relu')(x)
    x = Convolution2DTranspose(512,3,padding='same',activation = 'relu')(x)
    x_join_3 = UpSampling2D((2,2))(x)
    
    if 3 in pass_levels:
        x = concatenate([x_join_3,x_split_3],axis=3)
    else:
        x = x_join_3
    x = Convolution2DTranspose(256,3,padding='same', activation = 'relu')(x)
    x = Convolution2DTranspose(256,3,padding='same', activation = 'relu')(x)
    x = Convolution2DTranspose(256,3,padding='same', activation = 'relu')(x)
    x_join_2 = UpSampling2D((2,2))(x)

    if 2 in pass_levels:
        x = concatenate([x_join_2,x_split_2],axis=3)
    else:
        x = x_join_2
    x = Convolution2DTranspose(128,3,padding='same', activation = 'relu')(x)
    x = Convolution2DTranspose(128,3,padding='same', activation = 'relu')(x)
    x = Convolution2DTranspose(128,3,padding='same', activation = 'relu')(x)
    x_join_1 = UpSampling2D((2,2))(x)

    if 1 in pass_levels:
        x = concatenate([x_join_1,x_split_1],axis =3)
    else:
        x = x_join_1
    x = Convolution2DTranspose(64,3,padding='same', activation = 'relu')(x)
    x = Convolution2DTranspose(64,3,padding='same', activation = 'relu')(x)
    x = Convolution2DTranspose(no_labels,1,padding='same',activation='softmax')(x)


    model = Model(img_input, x)
    model.layers[1].trainable = False
    for i in range(14):
        (model.layers[i+2]).set_weights((bottle_neck.layers[i+1]).get_weights())
        (model.layers[i+2]).trainable = False

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = [metrics.categorical_accuracy])
    return model

