# -*- coding: utf-8 -*-
# Author : Ali Mirzaei
# Date : 05/10/2017
# Semi-Supervised Adverserial Autoencoder Fig II in paper

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape, concatenate
from keras.datasets import mnist
from keras.optimizers import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt


class AAN():
    def __init__(self, img_shape=(28, 28), encoded_dim=2):
        self.encoded_dim = encoded_dim
        self.optimizer_reconst = Adam(0.0001)
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
        decoder = Sequential()
        decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim))
        decoder.add(Dense(1000, activation='relu'))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        decoder.summary()
        return decoder

    def _getDescriminator(self, input_dim):
        """ Build Descriminator Model Based on Paper Configuration
        Args:
            input_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        latent_input = Input(shape=(input_dim,))
        labels_input = Input(shape=(11,))
        concated = concatenate([latent_input, labels_input])
        discriminator = Sequential()
        discriminator.add(Dense(1000, activation='relu',
                                input_dim=input_dim+11))
        discriminator.add(Dense(1000, activation='relu'))
        discriminator.add(Dense(1, activation='sigmoid'))
        out = discriminator(concated)
        discriminator_model = Model([latent_input, labels_input], out)
        discriminator_model.summary()
        return discriminator_model

    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        self.discriminator = self._getDescriminator(encoded_dim)
        img = Input(shape=img_shape)
        label_code = Input(shape=(11,))
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        valid = self.discriminator([encoded_repr, label_code])
        self.encoder_discriminator = Model([img, label_code], valid)
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
        (latent, labels) = self.generateRandomVectors(range(10))
        imgs = self.decoder.predict(latent)
        for index, img in enumerate(imgs):
            img = img.reshape((28, 28))
            ax = fig.add_subplot(1, 10, index+1)
            ax.set_axis_off()
            ax.imshow(img, cmap="gray")
        fig.savefig(str(epochnumber)+".png")
        plt.show()
        plt.close(fig)

    def train(self, x_train, batch_size=32, epochs=5000, save_image_interval=50):
        save_interval = 50
        for epoch in range(epochs):
            #---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            y_batch = y_train[idx]
            # Generate a half batch of new images
            latent_fake = self.encoder.predict(imgs)
            #gen_imgs = self.decoder.predict(latent_fake)
            #latent_real = np.random.normal(size=(half_batch, self.encoded_dim))
            (latent_real, labels) = self.generateRandomVectors(y_batch)
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([latent_real, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([latent_fake, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #idx = np.random.randint(0, x_train.shape[0], batch_size)
            #imgs = x_train[idx]
            #y_batch = y_train[idx]
            #(_, labels) = self.generateRandomVectors(y_batch)
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))

            # Train the autoencode reconstruction
            g_loss_reconstruction = self.autoencoder.train_on_batch(imgs, imgs)

            # Train generator
            g_logg_similarity = self.encoder_discriminator.train_on_batch([imgs, labels], valid_y)
            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1],
                   g_logg_similarity[1], g_loss_reconstruction))
            if(epoch % save_interval == 0):
                self.imagegrid(epoch)

    def generateRandomVectors(self, y_train):
        vectors = []
        labels = np.zeros((len(y_train), 11))
        labels[range(len(y_train)), np.array(y_train).astype(int)] = 1
        for index,y in enumerate(y_train):
            if y == 10:
                l = np.random.randint(0, 10)
            else:
                l = y
            mean = [10*np.sin((l*2*np.pi)/10), 10*np.cos((l*2*np.pi)/10)]
            vec = np.random.multivariate_normal(mean=mean, cov=np.eye(2),
                                                size=1)
            vectors.append(vec)
        return (np.array(vectors).reshape(-1, 2), labels)

if __name__ == '__main__':
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    idx_unlabel = np.random.randint(0, x_train.shape[0], 40000)
    y_train[idx_unlabel] = 10
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    ann = AAN()
    vecs,b = ann.generateRandomVectors(100*range(10))
    plt.scatter(vecs[:,0], vecs[:,1])

    ann.train(x_train)
    lat = ann.encoder.predict(x_train)
    plt.scatter(lat[:,0], lat[:,1], c=y_train)
