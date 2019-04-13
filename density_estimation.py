#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""

from __future__ import print_function

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from samplers import *

cuda = torch.cuda.is_available()

device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x * 2 + 1) + x * 0.75
d = lambda x: (1 - torch.tanh(x * 2 + 1) ** 2) * 2 + 0.75  # closures
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5, 5)
# exact
xx = np.linspace(-5, 5, 1000)
N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

# Jensen Shannon Divergence Loss Function
class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, d_x, d_y):
        return -(math.log(2.0) + 0.5 * (torch.mean(torch.log(d_x)) + torch.mean(torch.log(1.0 - d_y))))


# Wasserstein Distance Loss Function
class WDLoss(nn.Module):
    def __init__(self, _lambda):
        super(WDLoss, self).__init__()
        self._lambda = _lambda

    def forward(self, t_x, t_y, t_z):
        return -(torch.mean(t_x) - torch.mean(t_y) + self._lambda * torch.mean(torch.norm(t_y.grad - 1, 2)))


# MLP for Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size, activation_func):
        super(Discriminator, self).__init__()
        layers = []
        hidden_size.append(1)  # Output of discriminator
        for hidden in hidden_size:
            layers += [nn.Linear(input_dim, hidden)]
            layers += [nn.ReLU()]
            input_dim = hidden
        layers[-1] = activation_func
        self.disc = nn.Sequential(*layers)

    def forward(self, x, y):
        d_x = self.disc(x)
        d_y = self.disc(y)
        return d_x, d_y


def iterate(epoch, model, criterion, optimizer, total_samples, x_sampler, y_sampler, theta, mode='train'):
    if mode == 'train':
        model.train()
    elif mode == 'valid':
        model.eval()

    run_loss = 0.
    num_samples = 0

    monitor = tqdm(range(total_samples), desc=mode)
    for _ in monitor:
        x_tensor = torch.Tensor(next(x_sampler)).to(device)
        y_tensor = torch.Tensor(next(y_sampler)).to(device)

        d_x, d_y = model(x_tensor, y_tensor)

        if isinstance(criterion, WDLoss):
            z_sampler = iter(distribution1(0))
            z_tensor = nn.Variable(torch.Tensor(next(z_sampler)[1]).to(device))
            loss = criterion(d_x, d_y, z_tensor)
        else:
            loss = criterion(d_x, d_y)

        run_loss += loss.item() * x_tensor.shape[0]
        num_samples += x_tensor.shape[0]

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        monitor.set_postfix(epoch=epoch, loss=run_loss / num_samples, theta=theta, b=x_tensor.shape[0])

    return run_loss


if __name__ == '__main__':
    # Question 1

    # Configuration
    learning_rate = 1e-3
    batch_size = 512
    hidden_size = [512, 512]


    total_samples = 4096
    epochs = 1

    thetas = np.arange(-1, 1.1, 0.1)

    # Init model
    activation = nn.Sigmoid()
    model = Discriminator(input_dim=2, hidden_size=hidden_size, activation_func=activation)
    criterion = JSDLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # CUDA
    model = model.to(device)

    jsd_losses = []
    for idx, theta in enumerate(thetas):

        # Samplers
        iter_dist_x = iter(distribution1(0, batch_size=batch_size))
        # Create a list of distribution from theta=-1 to +1
        iter_dist_y = iter(distribution1(theta, batch_size=batch_size))

        jsd_losses.append([])
        for epoch in range(epochs):
            train_loss = iterate(epoch, model, criterion, optimizer, total_samples, iter_dist_x, iter_dist_y, theta,
                                 mode='train')
            valid_loss = iterate(epoch, model, criterion, optimizer, 32, iter_dist_x, iter_dist_y, theta,
                                 mode='valid')
            jsd_losses[idx].append(train_loss)

    # Question 2

    criterion = WDLoss(10)
    wd_losses = []
    for idx, theta in enumerate(thetas):
        # Samplers
        iter_dist_x = iter(distribution1(0, batch_size=batch_size))
        # Create a list of distribution from theta=-1 to +1
        iter_dist_y = iter(distribution1(theta, batch_size=batch_size))

        wd_losses.append([])
        for epoch in range(epochs):
            train_loss = iterate(epoch, model, criterion, optimizer, total_samples, iter_dist_x, iter_dist_y,
                                 mode='valid')
            jsd_losses[idx].append(train_loss)

    print(jsd_losses)
    print()
    print(wd_losses)

############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density


r = xx  # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(xx, r)
plt.title(r'$D(x)$')

estimate = np.ones_like(xx) * 0.2  # estimate the density of distribution4 (on xx) using the discriminator;
# replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1, 2, 2)
plt.plot(xx, estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
plt.legend(['Estimated', 'True'])
plt.title('Estimated vs True')
