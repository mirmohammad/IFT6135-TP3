#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""

from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from samplers import *

cuda = torch.cuda.is_available()

device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')


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
        return -(torch.mean(t_x) - torch.mean(t_y) - self._lambda * torch.mean((torch.norm(t_z, dim=1) - 1).pow(2)))


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

    def reset_parameters(self):
        for layer in self.disc:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        d_x = self.disc(x)
        return d_x


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

        d_x = model(x_tensor)
        d_y = model(y_tensor)

        if isinstance(criterion, WDLoss):
            z_sampler = iter(distribution1(0))
            z_tensor = Variable(torch.Tensor(next(z_sampler)), requires_grad=True).to(device)
            d_z = model(z_tensor)
            gradients = torch.autograd.grad(outputs=d_z, inputs=z_tensor, grad_outputs=torch.ones(d_z.size()).cuda(),
                                            create_graph=True, retain_graph=True)[0]
            loss = criterion(d_x, d_y, gradients)
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
    batch_size = 64
    hidden_size = [512, 512]

    total_samples = 2048
    epochs = 100

    thetas = np.arange(-1, 1.1, 0.1)

    # Init model
    activation = nn.ReLU()
    model = Discriminator(input_dim=2, hidden_size=hidden_size, activation_func=activation)
    # criterion = JSDLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # CUDA
    model = model.to(device)

    # Question 1

    # jsd_losses = []
    # for idx, theta in enumerate(thetas):
    #
    #     model.reset_parameters()
    #     optimizer.zero_grad()
    #
    #     # Samplers
    #     iter_dist_x = iter(distribution1(0, batch_size=batch_size))
    #     # Create a list of distribution from theta=-1 to +1
    #     iter_dist_y = iter(distribution1(theta, batch_size=batch_size))
    #
    #     jsd_losses.append([])
    #     for epoch in range(epochs):
    #         train_loss = iterate(epoch, model, criterion, optimizer, total_samples, iter_dist_x, iter_dist_y, theta,
    #                              mode='train')
    #         valid_loss = iterate(epoch, model, criterion, optimizer, 32, iter_dist_x, iter_dist_y, theta,
    #                              mode='valid')
    #         jsd_losses[idx].append(train_loss)

    # Question 2

    criterion = WDLoss(10)
    wd_losses = []
    for idx, theta in enumerate(thetas):

        model.reset_parameters()
        optimizer.zero_grad()

        # Samplers
        iter_dist_x = iter(distribution1(0, batch_size=batch_size))
        # Create a list of distribution from theta=-1 to +1
        iter_dist_y = iter(distribution1(theta, batch_size=batch_size))

        wd_losses.append([])
        for epoch in range(epochs):
            train_loss = iterate(epoch, model, criterion, optimizer, total_samples, iter_dist_x, iter_dist_y, theta,
                                 mode='train')
            valid_loss = iterate(epoch, model, criterion, optimizer, 32, iter_dist_x, iter_dist_y, theta,
                                 mode='valid')
            wd_losses[idx].append(train_loss)
