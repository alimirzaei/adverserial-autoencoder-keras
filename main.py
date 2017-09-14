# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

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
encoder.add(Dense(encoded_dim, activation='relu'))

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

autoencoder = Model(img, [encoded_repr, gen_img])


# Discriminator
discriminator = Sequential()
discriminator.add(Dense(1000, activation='relu', input_dim=encoded_dim))
discriminator.add(Dense(1000, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()
valid = discriminator(encoded_repr)
adversarial_autoencoder = Model(img, [gen_img, valid])

adversarial_autoencoder.compile(optimizer=optimizer, loss=['mse','binary_crossentropy'], loss_weights=[0.999, 0.001])
discriminator.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])

batch_size = 32
epochs = 20000
half_batch = int(batch_size / 2)

for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random half batch of images
    idx = np.random.randint(0, x_train.shape[0], half_batch)
    imgs = x_train[idx]

    # Generate a half batch of new images
    latent_fake, gen_imgs = autoencoder.predict(imgs)
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
    g_loss = adversarial_autoencoder.train_on_batch(imgs, [imgs, valid_y])

    # Plot the progress
    print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

#    # If at save interval => save generated image samples
#    if epoch % save_interval == 0:
#        # Select a random half batch of images
#        idx = np.random.randint(0, X_train.shape[0], 25)
#        imgs = X_train[idx]
#        self.save_imgs(epoch, imgs)
#
#autoencoder.fit(x_train, x_train, epochs=10, batch_size=256)

_ , decoded_imgs = autoencoder.predict(x_test)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
