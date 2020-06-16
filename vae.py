# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

import os

import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine

from torch.utils.data import TensorDataset, DataLoader

MAX_SEQ_LEN = 64
MAX_VEL = 127.
PITCH_DIM = 88
DATA_DIM = MAX_SEQ_LEN*PITCH_DIM
EPS = 1e-8

# for loading and batching music dataset
def setup_data_loaders(train_data_path, test_data_path, batch_size=16, use_cuda=False):
    train_data = torch.tensor(
        np.load(train_data_path) / MAX_VEL + EPS, 
        dtype=torch.float)
    train_data = torch.clamp(train_data, min=0., max=1.)
    train_data = train_data.view(-1, MAX_SEQ_LEN*PITCH_DIM)
    train_set = TensorDataset(train_data)

    test_data = torch.tensor(
        np.load(test_data_path) / MAX_VEL + EPS, 
        dtype=torch.float)
    test_data = test_data.view(-1, MAX_SEQ_LEN*PITCH_DIM)
    train_data = torch.clamp(test_data, min=0., max=1.)
    test_set = TensorDataset(test_data)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(DATA_DIM, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, DATA_DIM)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, DATA_DIM)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x DATA_DIM
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img

def model(self, x):
    # register PyTorch module `decoder` with Pyro
    pyro.module("decoder", self.decoder)
    with pyro.plate("data", x.shape[0]):
        # setup hyperparameters for prior p(z)
        z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
        z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        # decode the latent code z
        loc_img = self.decoder.forward(z)
        # score against actual images
        pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), 
                    obs=x.reshape(-1, DATA_DIM))

# define the guide (i.e. variational distribution) q(z|x)
def guide(self, x):
    # register PyTorch module `encoder` with Pyro
    pyro.module("encoder", self.encoder)
    with pyro.plate("data", x.shape[0]):
        # use the encoder to get the parameters used to define q(z|x)
        z_loc, z_scale = self.encoder.forward(x)
        # sample the latent code z
        pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim) # TODO: adding conditional
        self.decoder = Decoder(z_dim, hidden_dim) # TODO: adding conditional

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, DATA_DIM))

            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

def sample_vae(vae, cuda):
    x = torch.zeros([1, DATA_DIM])
    if cuda:
        x = x.cuda()

    for i in range(10):
        samples = []
        for rr in range(100):
            # get loc from the model
            sample_loc_i = vae.model(x)
            sampled_data = sample_loc_i[0].view(1, MAX_SEQ_LEN, PITCH_DIM).cpu().data.numpy() * MAX_VEL
            samples.append(sampled_data)
    
    return samples

def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    normalizer_train = len(train_loader.dataset)

    for x in train_loader:
        # if on GPU put mini-batch into CUDA memory
        x = x[0]
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        step_value = svi.step(x)        
        epoch_loss += step_value/ normalizer_train

    # return epoch loss    
    total_epoch_loss_train = epoch_loss 
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    normalizer_test = len(test_loader.dataset)
    # compute the loss over the entire test set
    for x in test_loader:
        x = x[0]
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x) / normalizer_test
    
    total_epoch_loss_test = test_loss
    return total_epoch_loss_test

def custom_elbo(model, guide, *args, **kwargs):
    # run the guide and trace its execution
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    # run the model and replay it against the samples from the guide
    model_trace = poutine.trace(
        poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

    # construct the elbo loss function    
    elbo = -1*(model_trace.log_prob_sum().cuda() - guide_trace.log_prob_sum().cuda())        

    return elbo

def main(args):
    # clear param store
    pyro.clear_param_store()
    pyro.enable_validation(True)
    pyro.distributions.enable_validation(False)
    pyro.set_rng_seed(0)

    # setup MNIST data loaders
    # train_loader, test_loader
    train_loader, test_loader = setup_data_loaders(args.train_data_path, args.test_data_path, batch_size=16, use_cuda=args.cuda)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)
    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=custom_elbo)
        
    train_elbo = []
    test_elbo = []
    # training loop

    for epoch in range(args.num_epochs):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=args.cuda)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=args.cuda)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

    samples = sample_vae(vae, args.cuda)
    samples = np.squeeze(samples)
    samples = np.array(np.round(samples, decimals=0), dtype=int)
    np.save(args.output_path + '/beethoven64_generated.npy', samples)

    return vae



if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--train_data_path', default='datasets/beethoven64_train.npy', type=str)
    parser.add_argument('--test_data_path', default='datasets/beethoven64_test.npy', type=str)
    parser.add_argument('--output_path', default='outputs', type=str)   
    parser.add_argument('-n', '--num-epochs', default=201, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=3.0e-5, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help='whether to use cuda')
    parser.add_argument('-i-tsne', '--tsne_iter', default=100, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()

    model = main(args)
    