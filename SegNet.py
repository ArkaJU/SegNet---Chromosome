from keras.layers import Input
from keras.layers.core import Activation, Reshape, Dropout
from keras.layers.convolutional import MaxPooling2D, UpSampling2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils

def to_categorical(y, nb_classes):
    num_samples = len(y)
    Y = np_utils.to_categorical(y.flatten(), nb_classes)
    return Y.reshape((num_samples, y.size // num_samples, nb_classes))

def SegNet(input_shape=(88, 88, 1), classes=4):

    img_input = Input(shape=input_shape)
    x = img_input

    # Encoder
    
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', 
                                              kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
                                              kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', 
                                               kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', 
                                               kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', 
                                               kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)


    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', 
                                               kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', 
                                               kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', 
                                               kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    #Decoder
    
    # Deconv Block 1
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, activation='relu', padding='same', 
                                                      kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=3, activation='relu', padding='same', 
                                                      kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=3, activation='relu', padding='same', 
                                                      kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Deconv Block 2
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same', 
                                                     kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same', 
                                                     kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(4, kernel_size=3, activation='relu', padding='same',
                                                    kernel_initializer = 'he_normal')(x)
    x = Dropout(0.25)(x)
    
    
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    
    model = Model(img_input, x)
    
    return model