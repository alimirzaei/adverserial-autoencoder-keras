# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# Author : Ali Mirzaei
# Date : 19/09/2017


from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.datasets import mnist, cifar10
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
import helpers
from sklearn.model_selection import GridSearchCV
import keras
from tqdm import tqdm

class GAE():
    def __init__(self, img_shape=(28, 28), encoded_dim=2):
        self.encoded_dim = encoded_dim
        self.optimizer = Adam(0.001)
        self._initAndCompileFullModel(img_shape, encoded_dim)
        self.img_shape = img_shape

    def _genEncoderModel(self, img_shape, encoded_dim):
        """ Build Encoder Model Based on Paper Configuration
        Args:
            img_shape (tuple) : shape of input image
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        encoder = Sequential()
        #encoder.add(Flatten(input_shape=img_shape))
        encoder.add(Conv2D(32, (5, 5), activation='tanh', input_shape=img_shape))
        encoder.add(Flatten())
        #encoder.add(Dense(100, activation='tanh'))
        encoder.add(Dense(encoded_dim, activation='tanh'))
        encoder.summary()
        return encoder

    def _getDecoderModel(self, encoded_dim, img_shape):
        """ Build Decoder Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
            img_shape (tuple) : shape of target images
        Return:
            A sequential keras model
        """
        decoder = Sequential()
        #decoder.add(Dense(100, activation='tanh', input_dim=encoded_dim))
        decoder.add(Dense((img_shape[0] - 2) * (img_shape[1] - 2) * 3, activation='tanh', input_dim=encoded_dim))
        decoder.add(Reshape(((img_shape[0] - 2), (img_shape[1] - 2), 3)))
        decoder.add(Conv2DTranspose(3, (3,3), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        decoder.summary()
        return decoder


    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        img = Input(shape=img_shape)
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        self.autoencoder.compile(optimizer=self.optimizer, loss='mse')
    def imagegrid(self, epochnumber):
        fig = plt.figure(figsize=[20, 20])
        for i in range(-5, 5):
            for j in range(-5,5):
                topred = np.array((i*0.5,j*0.5))
                topred = topred.reshape((1, 2))
                img = self.decoder.predict(topred)
                img = img.reshape((28, 28))
                ax = fig.add_subplot(10, 10, (i+5)*10+j+5+1)
                ax.set_axis_off()
                ax.imshow(img, cmap="gray")
        fig.savefig(str(epochnumber)+".png")
        plt.show()
        plt.close(fig)

    def train(self, x_train, batch_size=32, epochs=5):
        self.autoencoder.load_weights('15619.hdf5')
        l=0
        tq = tqdm(range(len(x_train)*epochs/batch_size))
        for i in tq:
            tq.set_description("MSE Loss (%f)" % l)
            idx= np.random.randint(0, len(x_train), batch_size)
            x_batch = x_train[idx]
            x_out = copy.copy(x_batch)
            mask = np.random.choice([0, 1], size=(batch_size, 32, 32, 1), p=[1./10, 9./10])
            mask = np.repeat(mask, 3, 3)
            mm = np.ma.masked_array(x_batch, mask= mask)
            x_batch[mm.mask]=0
            l = self.autoencoder.train_on_batch(x_batch, x_out)
            if(i%(len(x_train)/batch_size)==(len(x_train)/batch_size)-1):
                self.autoencoder.save_weights(str(i)+'.hdf5')
#                                 callbacks=[keras.callbacks.ModelCheckpoint('weights_cifar.{epoch:02d}.hdf5', 
#                                                                           verbose=0, 
#                                                                           save_best_only=False, 
#                                                                           save_weights_only=False, 
#                                                                           mode='auto', 
#                                                                           period=1)])
#        codes = self.encoder.predict(x_train)
#        params = {'bandwidth': [3.16]}#np.logspace(0, 2, 5)}
#        grid = GridSearchCV(KernelDensity(), params, n_jobs=4)
#        grid.fit(codes)
#        print grid.best_params_
#        self.kde = grid.best_estimator_
#        self.kde = KernelDensity(kernel='gaussian', bandwidth=.2).fit(codes)

    def generate(self, n = 10000):
        codes = self.kde.sample(n)
        images = self.decoder.predict(codes)
        return images

    def generateAndPlot(self, x_test):
        fig = plt.figure(figsize=[20, 20])
        images = x_test
        index = 1
        reconstructed_image = self.autoencoder.predict(x_test)
        for image,r in zip(images, reconstructed_image):
            image = image.reshape(self.img_shape)
            ax = fig.add_subplot(len(x_test), 2, index)
            index=index+1
            ax.set_axis_off()
            ax.imshow(image, cmap="gray")
            ax = fig.add_subplot(len(x_test), 2, index)
            index=index+1
            ax.set_axis_off()
            ax.imshow(r, cmap="gray")
            
        fig.savefig("cfar10_generated.png")
        plt.show()

    def meanLogLikelihood(self, x_test):
        KernelDensity(kernel='gaussian', bandwidth=0.2).fit(codes)


def plotResults(ann, x_test, key_numbers = 10):
    fig = plt.figure(figsize=(10, 10*len(x_test)/4))
    for index, x in enumerate(x_test):
        print('Processing Image ', index)
        selected_pixels = []
        best_partial_x = np.zeros(x.shape)
        for p in range(key_numbers):
            errors = []
            partial_images = []
            for pixel in range(32*32):
                partial_x = copy.copy(best_partial_x)
                partial_x[pixel/32, pixel%32,:] = x[pixel/32, pixel%32,:]
                partial_images.append(copy.copy(partial_x))
            partial_images= np.array(partial_images)
            y = ann.autoencoder.predict(partial_images)
            errors = np.sum(np.abs(y-x), axis=(1,2,3))
            best_pixel = np.argmin(errors)
            best_partial_x[best_pixel/32, best_pixel%32,:] = x[best_pixel/32, best_pixel%32,:]
            selected_pixels.append(best_pixel)
        partial_x = np.zeros(x.shape)
        mask = np.zeros(x.shape)
        for pixel in selected_pixels:
            partial_x[pixel/32, pixel%32,:] = x[pixel/32, pixel%32,:]
            mask[pixel/32, pixel%32] = 1
        y = ann.autoencoder.predict(partial_x.reshape(1,32,32,3))
        ax = fig.add_subplot(len(x_test),4,index*4+1)
        ax.imshow(x)
        ax = fig.add_subplot(len(x_test),4,index*4+2)
        ax.imshow(mask)
        ax = fig.add_subplot(len(x_test),4,index*4+3)
        ax.imshow(partial_x)
        ax = fig.add_subplot(len(x_test),4,index*4+4)
        ax.imshow(y[0])
        plt.show()
    fig.savefig(str(key_numbers)+'cifar3.jpg')


if __name__ == '__main__':
    # Load MNIST dataset
    import copy
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
#    x_out = copy.copy(x_train)
#    x_in = []
#    for x in x_train:
#        mask = np.random.choice([0, 1], size=(32, 32), p=[1./10, 9./10])
#        mask = np.dstack([mask, mask, mask])
#        mm = np.ma.masked_array(x, mask= mask)
#        x[mm.mask]=0
#        x_in.append(x)
#    x_in = np.array(x_in)
    ann = GAE(img_shape = x_train[0].shape, encoded_dim=100)
    #ann.autoencoder.load_weights('weights.03.hdf5')
    ann.train(x_train, epochs=0)
    #ann.generateAndPlot(x_in[0:10])
    plotResults(ann, x_test[np.random.randint(0, len(x_test), 5)],100)
    #generated = ann.generate(10000)
    #L = helpers.approximateLogLiklihood(generated, x_test, searchSpace=[.1])
    #print L
    #codes = ann.kde.sample(1000)
    #ax = Axes3D(plt.gcf())
    #codes = ann.encoder.predict(x_train)
    #plt.scatter(codes[:,0], codes[:,1], c=y_train)
    #plt.show()
