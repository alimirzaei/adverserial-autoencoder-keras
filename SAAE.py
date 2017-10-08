# -*- coding: utf-8 -*-
# Author : Ali Mirzaei
# Date : 06/10/2017

# Supervised Adverserial Autoencoder Fig 6

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape, concatenate
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


class SAAE():
    def __init__(self, img_shape=(28, 28), encoded_dim=2):
        self.encoded_dim = encoded_dim
        self.optimizer_reconst = Adam(0.001)
        self.optimizer_discriminator = Adam(0.0001)
        self._initAndCompileFullModel(img_shape, encoded_dim)

    def _genEncoderModel(self, img_shape, encoded_dim):
        """ Build Encoder Model Based on Paper Configuration
        Args:
            img_shape (tuple) : shape of input image
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        encoder = Sequential()
        encoder.add(Flatten(input_shape=img_shape))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(encoded_dim))
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
        latent_input = Input(shape=(encoded_dim,))
        labels_input = Input(shape=(10,))
        concated = concatenate([latent_input, labels_input])
        decoder = Sequential()
        decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim+10))
        decoder.add(Dense(1000, activation='relu'))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        out = decoder(concated)
        decoder_model = Model([latent_input, labels_input], out)
        decoder_model.summary()
        return decoder_model

    def _getDescriminator(self, encoded_dim):
        """ Build Descriminator Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        discriminator = Sequential()
        discriminator.add(Dense(1000, activation='relu',
                                input_dim=encoded_dim))
        discriminator.add(Dense(1000, activation='relu'))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.summary()
        return discriminator

    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        self.discriminator = self._getDescriminator(encoded_dim)
        img = Input(shape=img_shape)
        label = Input(shape=(10,))
        encoded_repr = self.encoder(img)
        gen_img = self.decoder([encoded_repr, label])
        self.autoencoder = Model([img, label], gen_img)
        valid = self.discriminator(encoded_repr)
        self.encoder_discriminator = Model(img, valid)
        self.discriminator.compile(optimizer=self.optimizer_discriminator,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
        self.autoencoder.compile(optimizer=self.optimizer_reconst,
                                 loss ='mse')
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.encoder_discriminator.compile(optimizer=self.optimizer_discriminator,
                                           loss='binary_crossentropy',
                                           metrics=['accuracy'])
    def imagegrid(self, epochnumber):
        fig = plt.figure(figsize=[20, 20])
        labels = self.getOneHotCodes(range(10))
        styles = np.random.normal(size=(10, self.encoded_dim))
        k=0
        for label in labels:
            for style in styles:
                img = self.decoder.predict([style.reshape(-1,self.encoded_dim), label.reshape(-1,10)])
                img = img.reshape((28, 28))
                k = k + 1
                ax = fig.add_subplot(10, 10, k)
                ax.set_axis_off()
                ax.imshow(img, cmap="gray")
        fig.savefig("images/SAAE/" + str(epochnumber) + ".png")
        plt.show()
        plt.close(fig)

    def getOneHotCodes(self, y_train):
        labels = np.zeros((len(y_train), 10))
        labels[range(len(y_train)), np.array(y_train).astype(int)] = 1
        return labels

    def train(self, x_train, y_train, batch_size=32, epochs=5000, save_image_interval=50):
        labels = self.getOneHotCodes(y_train)
        save_interval = 50
        for epoch in range(epochs):
            #---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            labels_batch = labels[idx, :]
            # Generate a half batch of new images
            latent_fake = self.encoder.predict(imgs)
            #gen_imgs = self.decoder.predict(latent_fake)
            latent_real = np.random.normal(size=(batch_size, self.encoded_dim))
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #idx = np.random.randint(0, x_train.shape[0], batch_size)
            #imgs = x_train[idx]
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))

            # Train the autoencode reconstruction
            g_loss_reconstruction = self.autoencoder.train_on_batch([imgs, labels_batch], imgs)

            # Train generator
            g_logg_similarity = self.encoder_discriminator.train_on_batch(imgs, valid_y)
            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1],
                   g_logg_similarity[1], g_loss_reconstruction))
            if(epoch % save_interval == 0):
                self.imagegrid(epoch)


if __name__ == '__main__':
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    ann = SAAE()
    ann.train(x_train, y_train)
