import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torchvision import datasets, transforms
from torch import optim
import matplotlib.pyplot as plt

kwargs =  {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist', train=True, download=True,
                   transform= transforms.ToTensor()),
    batch_size=32, shuffle=True, **kwargs)

class GAE(nn.Module):
    def __init__(self, input_shape=(28, 28), latent_dim=2):
        super(GAE, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(np.prod(input_shape), 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, latent_dim)
        self.fc4 = nn.Linear(latent_dim, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, np.prod(input_shape))
    def forward(self, x):
        x = x.view((-1, 1, np.prod(self.input_shape)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        x = x.view((-1, 1)+self.input_shape)
        return x
    def generateAndPlot(self, x_test, n = 10, fileName="generated.png"):
        fig = plt.figure(figsize=[20, 20*n/3])
        for i in range(n):
            x_in = x_test[np.random.randint(len(x_test))]
            x=copy.copy(x_in)
            mask = np.random.choice([0, 1], size=(28, 28), p=[1./10, 9./10])
            mm = np.ma.masked_array(x, mask= mask)
            x[mm.mask]=0
            y = self.forward(Variable(torch.from_numpy(x.reshape(1, 1, 28, 28))))
            ax = fig.add_subplot(n, 3, i*3+1)
            ax.set_axis_off()
            ax.imshow(x)
            ax = fig.add_subplot(n, 3, i*3+2)
            ax.set_axis_off()
            ax.imshow(y[0][0].data.numpy())
            ax = fig.add_subplot(n, 3, i*3+3)
            ax.set_axis_off()
            ax.imshow(x_in)
        fig.savefig(fileName)
        plt.show()

class KeyPointNetwork(nn.Module):
    def __init__(self, input_shape=(28, 28)):
        super(KeyPointNetwork, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.input_shape = input_shape
    def forward(self, x):
        x = x.view(-1, 1, np.prod(self.input_shape))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))*np.prod(self.input_shape)
        return x

class ImageCompletion(nn.Module):
    def __init__(self, input_shape=(28,28), generatorModel = 'gae.model'):
        super(ImageCompletion, self).__init__()
        self.gae = GAE(input_shape=input_shape)
        self.gae = torch.load(generatorModel)
        for p in self.gae.parameters():
            p.require_grad=False
        self.kpn = KeyPointNetwork(input_shape=input_shape)
        self.input_shape = input_shape
    def forward(self, x):
        keys = self.kpn(x)
        in_images = torch.zeros(x.size())
        for i in range(32):
            im = x[i, :, :]
            for key in keys[i,0].data.numpy():
                key = int(key)
                a = in_images[i,key/28, key%28]
                b=im[key/28, key%28].data
#            sel_key  = keys.view(32, 1, 2, -1)[i, 0, :, :]
#            sel_key = sel_key.type(torch.LongTensor)
#            sel_image = in_images[i, 0, :, :]
#            sel_image[sel_key.data.numpy()] = im.data.index(sel_key.data.numpy())
        # in_images[keys.int().data.numpy()] = x.data.index(keys.int().data.numpy())
        # in_images[keys] = x[keys]
        y = self.gae(Variable(in_images))
        return y
    def showResults(self, x_test, n=10, fileName ="Keys.jpg"):
        import copy
        fig = plt.figure(figsize=[20, 20*n/3])
        for i in range(n):
            x_in = x_test[np.random.randint(len(x_test))]
            x=copy.copy(x_in)
            mask = self.kpn.forward(Variable(torch.from_numpy(x.reshape((1,1)+self.input_shape))))
            mm = np.zeros(self.input_shape)
            for element in mask[0,0].data.numpy():
                mm[int(element/self.input_shape[0]),int(element%self.input_shape[0])]=1
            ax = fig.add_subplot(n, 4, i*3+1)
            ax.set_axis_off()
            ax.imshow(x)
            ax = fig.add_subplot(n, 4, i*3+2)
            ax.set_axis_off()
            ax.imshow(mm)
            y = self.forward(Variable(torch.from_numpy(x.reshape((1,1)+self.input_shape))))
            ax = fig.add_subplot(n, 4, i*3+3)
            ax.set_axis_off()
            ax.imshow(y[0,0].data.numpy())
            ax = fig.add_subplot(n, 4, i*3+4)
            ax.set_axis_off()
            ax.imshow(x_in)
        fig.savefig(fileName)
        plt.show()

def trainGAE():
    import copy
    from keras.datasets import mnist
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    x_out = copy.copy(x_train)
    x_in = []
    for x in x_train:
        mask = np.random.choice([0, 1], size=(28, 28), p=[1./10, 9./10])
        mm = np.ma.masked_array(x, mask=mask)
        x[mm.mask] = 0
        x_in.append(x)
    x_in = np.array(x_in)
    model = GAE(latent_dim=6)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    model.train()
    batch_size = 32
    epochs = 1000
    torch.load('gae.model')
    for epoch in range(epochs):
        idx = np.random.randint(0, x_in.shape[0], batch_size)
        input_images = x_in[idx]
        output_images = x_out[idx]
        input_images, output_images = Variable(torch.from_numpy(input_images)), Variable(torch.from_numpy(output_images))
        optimizer.zero_grad()
        output = model.forward(input_images)
        l = loss(output, output_images)
        l.backward()
        optimizer.step()
        print(l.data.numpy()[0], epoch*100/epochs)
    model.generateAndPlot(x_test)
    torch.save(model, 'gae.model')

def trainKeyPoint():
    from keras.datasets import mnist
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    model = ImageCompletion()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    model.train()
    batch_size = 32
    epochs = 10
    model = torch.load('keypoint.model')
    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        input_images = x_train[idx]
        input_images = Variable(torch.from_numpy(input_images))
        optimizer.zero_grad()
        output = model.forward(input_images)
        l = loss(output, input_images)
        l.backward()
        optimizer.step()
        print(l.data.numpy()[0], epoch*100/epochs)
    model.showResults(x_test)
    torch.save(model, 'keypoint.model')
    
if __name__ == '__main__':
    trainKeyPoint()
