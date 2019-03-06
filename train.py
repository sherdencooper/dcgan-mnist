from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.core import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import keras.backend as K
import tensorflow as tf
from model import *
from PIL import Image
import math

BATCH_SIZE = 256


#prepare for the training data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

d = discriminator()
g = generator()
dcgan = GAN(g,d)
d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g.compile(loss='binary_crossentropy', optimizer="SGD")
dcgan.compile(loss='binary_crossentropy', optimizer=g_optim)
d.trainable = True
d.compile(loss='binary_crossentropy', optimizer=d_optim)


noise = np.zeros((BATCH_SIZE, 100))


for epoch in range(200):
    for index in range(int(X_train.shape[0]/BATCH_SIZE)):
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.normal(0, 0.5, 100)   #initialize the noise
        image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        generated_images = g.predict(noise)
        # X = np.concatenate((image_batch, generated_images))
        # y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
        if index % 2 == 0:
            X = image_batch
            y = [1] * BATCH_SIZE
        else:
            X = generated_images
            y = [0] * BATCH_SIZE
        d_loss = d.train_on_batch(X, y)
        print("epoch %d batch %d d_loss : %f" % (epoch, index, d_loss))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.normal(0, 0.5, 100)
        discriminator.trainable = False
        g_loss = dcgan.train_on_batch(noise, [1] * BATCH_SIZE)
        discriminator.trainable = True
        print("epoch %d batch %d g_loss : %f" % (epoch, index, g_loss))
        if index % 10 == 9:
            g.save_weights('generator', True)
            d.save_weights('discriminator', True)



def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


#generate images after training
def generate_images(BATCH_SIZE, nice=False):
    g = generator()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


generate_images(BATCH_SIZE)





