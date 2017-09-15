# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def imagegrid(dec, epochnumber):
    fig = plt.figure(figsize=[20, 20])
    for i in range(-5, 5):
        for j in range(-5,5):
            topred = np.array((i*0.5,j*0.5))
            topred = topred.reshape((1, 2))
            img = dec.predict(topred)
            img = img.reshape((28, 28))
            ax = fig.add_subplot(10, 10, (i+5)*10+j+5+1)
            ax.set_axis_off()
            ax.imshow(img, cmap="gray")
    fig.savefig(str(epochnumber)+".png")
    plt.show()
    plt.close(fig)

optimizer = Adam(0.0002, 0.5)
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

img_shape = (28, 28)
encoded_dim = 2

# build encoder

encoder = Sequential()
encoder.add(Flatten(input_shape=img_shape))
encoder.add(Dense(1000, activation='relu'))
encoder.add(Dense(1000, activation='relu'))
encoder.add(Dense(encoded_dim))

encoder.summary()

img = Input(shape=img_shape)
encoded_repr = encoder(img)

# Decoder
decoder = Sequential()

decoder.add(Dense(1000,activation='relu', input_dim=encoded_dim))
decoder.add(Dense(1000,activation='relu'))
decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
decoder.add(Reshape(img_shape))

decoder.summary()

gen_img = decoder(encoded_repr)

autoencoder = Model(img, gen_img)


# Discriminator
discriminator = Sequential()
discriminator.add(Dense(1000, activation='relu', input_dim=encoded_dim))
discriminator.add(Dense(1000, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()
valid = discriminator(encoded_repr)
encoder_discriminator = Model(img, valid)


discriminator.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
autoencoder.compile(optimizer=optimizer, loss ='mse')
for layer in discriminator.layers:
    layer.trainable=False
encoder_discriminator.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])

batch_size = 32
epochs = 30000
half_batch = int(batch_size / 2)
save_interval = 500
for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random half batch of images
    idx = np.random.randint(0, x_train.shape[0], half_batch)
    imgs = x_train[idx]

    # Generate a half batch of new images
    latent_fake = encoder.predict(imgs)
    gen_imgs = decoder.predict(latent_fake)
    latent_real = np.random.normal(size=(half_batch, encoded_dim))

    valid = np.ones((half_batch, 1))
    fake = np.zeros((half_batch, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(latent_real, valid)
    d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


    # ---------------------
    #  Train Generator
    # ---------------------

    # Select a random half batch of images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs = x_train[idx]

    # Generator wants the discriminator to label the generated representations as valid
    valid_y = np.ones((batch_size, 1))

    # Train the generator
    g_loss_reconstruction = autoencoder.train_on_batch(imgs, imgs)
    g_logg_similarity = encoder_discriminator.train_on_batch(imgs, valid_y)
    # Plot the progress
    print ("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1],
           g_logg_similarity[1], g_loss_reconstruction))
    if(epoch % save_interval == 0):
        imagegrid(decoder,epoch)
#    # If at save interval => save generated image samples
#    if epoch % save_interval == 0:
#        # Select a random half batch of images
#        idx = np.random.randint(0, X_train.shape[0], 25)
#        imgs = X_train[idx]
#        self.save_imgs(epoch, imgs)
#
#autoencoder.fit(x_train, x_train, epochs=10, batch_size=256)


