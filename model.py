from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.core import Flatten
from keras.layers import Dropout



def generator(input_dim=100):
    dropout = 0.4
    dim = 7
    depth = 64 * 4
    model = Sequential()
    # In: 100
    # Out: dim x dim x depth
    model.add(Dense(dim * dim * depth, input_dim=100))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Reshape((dim, dim, depth)))
    model.add(Dropout(dropout))

    # In: dim x dim x depth
    # Out: 2*dim x 2*dim x depth/2
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
    model.add(Conv2DTranspose(1, 5, padding='same'))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model


def discriminator(input_shape=(28, 28, 1),nb_filter=64):
    model = Sequential()
    dropout = 0.4
    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same',input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(2 * nb_filter, (5, 5), strides=(2, 2), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(4 * nb_filter, (5, 5), strides=(2, 2), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(8 * nb_filter, (5, 5), strides=(1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model

def GAN(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

