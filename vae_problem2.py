import os

import numpy as np
import torch
import torch.utils.data as data_utils
from torch import nn
from torch.autograd import Variable
from torch.functional import F
from torch.nn.modules import upsampling
from torch.optim import Adam
from torchvision.datasets import utils


def get_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        # utils.download_url(URL + filename, dataset_location, filename, None)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        dataset = data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata


train_data, valid_data, test_data = get_data_loader("binarized_mnist", 64)


# idea taken from https://github.com/sksq96/pytorch-vae/blob/master/vae.ipynb
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class VAE(nn.Module):
    def __init__(self, L=100, M=64, K=1, D=784):
        super(VAE, self).__init__()

        self.K = K
        self.M = M
        self.L = L
        self.D = D

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(64, 256, 5),
            nn.ELU()
        )

        self.enc2 = nn.Linear(256, self.L*2)

        # Decoder
        self.dec1 = nn.Linear(self.L, 256)

        self.dec2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(256, 64, 5, padding=4),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=2),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, 3, padding=2),
        )

    def reparametrizationTrick(self, mu, log_var):
        # Inspired from https://github.com/sksq96/pytorch-vae/blob/master/vae.ipynb
        # and : https://github.com/atinghosh/VAE-pytorch/blob/master/VAE_CNN_BCEloss.py
        #samples = []
        #for _ in range(self.K):
        esp = to_var(torch.randn(*mu.size()))
        std = torch.exp(log_var / 2)
        return mu + std*esp
        #sample = mu + std * esp
            #samples.append(sample)
        #return samples

    def encode(self, x):
        # print("Input encoder:")
        # print(x.shape)
        x = x.view(-1, 1, 28, 28)
        h = self.enc1(x)
        h = self.enc2(h.view(-1, 256))
        # print("output encoder:")
        # print(h.shape)
        mu, log_var = torch.chunk(h, 2, dim=1)
        # print("mu:")
        # print(mu.shape)
        # print("sigma:")
        # print(log_var.shape)
        return mu, log_var

    def decode(self, z):
        # Issue here TODO
        # print(z.shape)
        out = self.dec1(z)
        # print(out.shape)
        out = F.sigmoid(self.dec2(out.view(-1, 256, 1, 1)))  # TODO check view
        # print(out.shape)
        return out  #.view(-1, self.D)

    def forward(self, x):
        mu, log_var = self.encode(x)
        #samples = self.reparametrizationTrick(mu, log_var)
        # print([self.decode(sample) for sample in samples])
        #return torch.stack([self.decode(sample) for sample in samples]), mu, log_var
        sample = self.reparametrizationTrick(mu, log_var)
        return self.decode(sample), mu, log_var


class ELBOLoss(nn.Module):
    def __init__(self, ):
        super(ELBOLoss, self).__init__()
        self.recons_loss = nn.BCELoss(reduction='sum')
        # KL_loss = 0.5 * torch.sum(log_var.exp() + mu ** 2 - 1. - log_var)

    def forward(self, reconstruction, x, mu, log_var):
        loss = -self.recons_loss(reconstruction, x)
        KL_loss = 0.5 * torch.sum(-1 - log_var + mu ** 2 + log_var.exp())
        return -(loss - KL_loss)


def train(model, trainloader, epoch, epochs, criterion):
    model.train()
    elbos = 0
    total = 0
    for input_image in trainloader:
        if cuda:
            input_image = input_image.cuda()

        reconstruction_image, mu, var = model(input_image)
        loss = criterion(reconstruction_image, input_image, mu, var)
        elbos += loss.item()
        total += 64
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if total % 100 == 0:
            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item() / 64))
    return elbos / total


def evaluate(model, dataset, criterion):
    with torch.no_grad():
        model.eval()
        total = 0.
        ELBO = 0.
        for input_image in dataset:
            if cuda:
                input_image = input_image.cuda()
            reconstruction_image, mu, var = model(input_image)
            total += 64
            ELBO += criterion(reconstruction_image, input_image, mu, var)
    return ELBO / total


from torchsummary import summary

latent_size = 100
model = VAE()
criterion = ELBOLoss()
params = model.parameters()

optimizer = Adam(params, lr=3e-4)

cuda = torch.cuda.is_available()

if cuda:
    model = model.cuda()

##summary(model, (1, 28, 28))

train_elbos = []
valid_elbos = []
epochs = 20
for epoch in range(epochs):
    train_elbos += [train(model, train_data, epoch, epochs, criterion)]
    valid_elbos += [evaluate(model, valid_data, criterion)]
    print("Validation ELBO:", valid_elbos[-1])

torch.save(model, "VAE_p2_1.pt")
print("Model Saved.")
