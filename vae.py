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
from pyro.infer.util import torch_item
from pyro.util import warn_if_nan
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine
from iwae import Trace_IWAE
import wandb

from torch.utils.data import TensorDataset, DataLoader, Dataset

MAX_SEQ_LEN = 64
MAX_VEL = 127.
PITCH_DIM = 90
DATA_DIM = MAX_SEQ_LEN*PITCH_DIM
EPS = 1e-8
START_PITCH = 88
END_PITCH = 89

# for loading and batching music dataset

class CondDataset(Dataset):
    def __init__(self, *tensors):        
        # temp_tensor = tensors[0].reshape(-1, MAX_SEQ_LEN*2, PITCH_DIM)
        temp_tensor = tensors[0]
        # [DATA_SIZE, MAX_SEQ_LEN*2, PITCH_DIM]
        self.cond_tensors = temp_tensor[:,:MAX_SEQ_LEN,:].reshape(-1, MAX_SEQ_LEN*PITCH_DIM)
        self.tensors = temp_tensor[:,MAX_SEQ_LEN:,:].reshape(-1, MAX_SEQ_LEN*PITCH_DIM)
        self.length = len(self.tensors)

    def __getitem__(self, idx):
        x = self.tensors[idx]
            
        cond = self.cond_tensors[idx]
        sample = {'x': x, 'cond': cond}

        return sample

    def __len__(self):
        return self.length

def setup_data_loaders(train_data_path, test_data_path, batch_size=16, use_cuda=False):
    device = 'cuda' if use_cuda else 'cpu'
    train_data = torch.tensor(
        np.load(train_data_path) / MAX_VEL, 
        dtype=torch.float, device=device)
    train_data = torch.clamp(train_data, min=0., max=1.)
    train_set = CondDataset(train_data)

    test_data = torch.tensor(
        np.load(test_data_path) / MAX_VEL, 
        dtype=torch.float, device=device)
    test_data = torch.clamp(test_data, min=0., max=1.)
    test_set = CondDataset(test_data)

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
        self.fc1 = nn.Linear(DATA_DIM + DATA_DIM, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x, cond):
        inputs = torch.cat([x, cond], 1)
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension        
        inputs = inputs.reshape(-1, DATA_DIM + DATA_DIM)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(inputs))
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
        self.fc1 = nn.Linear(z_dim + DATA_DIM, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, DATA_DIM)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, cond):
        # define the forward computation on the latent z
        # first compute the hidden units
        inputs = torch.cat([z, cond], 1)
        hidden = self.softplus(self.fc1(inputs))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x DATA_DIM
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img

def model(self, x, cond):
    # register PyTorch module `decoder` with Pyro
    pyro.module("decoder", self.decoder)
    with pyro.plate("data", x.shape[0]):
        # setup hyperparameters for prior p(z)
        z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
        z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        # decode the latent code z
        loc_img = self.decoder.forward(z, cond)
        # score against actual images
        pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), 
                    obs=x.reshape(-1, DATA_DIM + DATA_DIM))

# define the guide (i.e. variational distribution) q(z|x)
def guide(self, x, cond):
    # register PyTorch module `encoder` with Pyro
    pyro.module("encoder", self.encoder)
    with pyro.plate("data", x.shape[0]):
        # use the encoder to get the parameters used to define q(z|x)
        z_loc, z_scale = self.encoder.forward(x, cond)
        # sample the latent code z
        pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 512-dimensional
    # and we use 1024 hidden units
    def __init__(self, z_dim=512, hidden_dim=1024, use_cuda=False):
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
    def model(self, x, cond):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z, cond)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, DATA_DIM))

            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, cond):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x, cond)
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
    # start = torch.cat(torch.zeros([1, MAX_SEQ_LEN, 88]) + torch.ones([1, MAX_SEQ_LEN, 1]) + torch.zeros([1, MAX_SEQ_LEN, 1]), dim=-1)
    start = torch.cat([torch.zeros(1, MAX_SEQ_LEN, 88), torch.ones(1, MAX_SEQ_LEN, 1), torch.zeros(1, MAX_SEQ_LEN, 1)], dim=-1)    

    if cuda:
        x = x.cuda()

    total_samples = []
    for i in range(10):
        samples = []
        cond = start.reshape(-1, MAX_SEQ_LEN * PITCH_DIM).cuda()
        for rr in range(20):
            # get loc from the model
            sample_loc_i = vae.model(x, cond)
            cond = sample_loc_i
            sampled_data = sample_loc_i[0].view(1, MAX_SEQ_LEN, PITCH_DIM).cpu().data.numpy() * MAX_VEL
            samples.append(sampled_data)
        total_samples.append(samples)
    
    return total_samples

def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    normalizer_train = len(train_loader.dataset)

    for data in train_loader:
        # if on GPU put mini-batch into CUDA memory
        cond = data['cond']
        x = data['x']
        if use_cuda:
            cond = cond.cuda()
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        step_value = svi.step(x, cond)
        epoch_loss += step_value/ normalizer_train

    # return epoch loss    
    total_epoch_loss_train = epoch_loss 
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    normalizer_test = len(test_loader.dataset)
    # compute the loss over the entire test set
    for data in test_loader:        
        cond = data['cond']
        x = data['x']
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            cond = cond.cuda()
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x, cond) / normalizer_test
    
    total_epoch_loss_test = test_loss
    return total_epoch_loss_test

# def IWAE_loss(self, model, guide, *args, **kwargs):
#     """
#     :returns: returns an estimate of the ELBO
#     :rtype: float

#     Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
#     """
#     elbo = 0.0
#     for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
#         elbo_particle = torch.exp(torch_item(model_trace.log_prob_sum())) - torch.exp(torch_item(guide_trace.log_prob_sum()))
#         elbo += elbo_particle / self.num_particles

#     loss = -elbo
#     warn_if_nan(loss, "loss")
#     return loss

def main(artist_name, args):
    # clear param store
    pyro.clear_param_store()
    pyro.enable_validation(True)
    pyro.distributions.enable_validation(False)
    pyro.set_rng_seed(0)
    # wandb.init(project="vae_midi", name="%s (num_particles=%d) [%s]"%(args.model_name, args.num_particles, artist_name))

    # setup MNIST data loaders
    # train_loader, test_loader
    train_loader, test_loader = setup_data_loaders(args.train_data_path, args.test_data_path, batch_size=32, use_cuda=args.cuda)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)
    # wandb.watch(vae)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    if args.model_name == 'VAE':
        loss = Trace_ELBO(num_particles=args.num_particles)
    elif args.model_name == 'IWAE':
        loss = Trace_IWAE(num_particles=args.num_particles)

    svi = SVI(vae.model, vae.guide, optimizer, loss=loss)
            
    train_elbo = []
    test_elbo = []
    min_valid_loss = 100000.
    # training loop

    for epoch in range(args.num_epochs):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=args.cuda)        
        train_elbo.append(-total_epoch_loss_train)
        # wandb.log({"Train ELBO": -total_epoch_loss_train})
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=args.cuda)
            test_elbo.append(-total_epoch_loss_test)
            # wandb.log({"Test ELBO": -total_epoch_loss_test})

            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

            if total_epoch_loss_test < min_valid_loss:
                min_valid_loss = total_epoch_loss_test
                torch.save(vae, args.output_path + 'best_model.pt')
        
    # Generate samples from best model
    vae_best = torch.load(args.output_path + 'best_model.pt')    
    samples = sample_vae(vae, args.cuda)  # [10, 20, MAX_SEQ_LEN, MAX_PITCH]
    samples = np.squeeze(samples)
    samples = np.array(np.round(samples, decimals=0), dtype=int)
    
    # import IPython; IPython.embed()
    train_elbo_ = np.concatenate([np.arange(0, args.num_epochs), np.array(train_elbo)])
    test_elbo_ = np.concatenate([np.arange(0, args.num_epochs, args.test_frequency), np.array(test_elbo)])

    np.save(args.output_path + 'generated_samples.npy', samples)
    np.save(args.output_path + 'train_elbo.npy', train_elbo_)
    np.save(args.output_path + 'test_elbo.npy', test_elbo_)
    
    return vae



if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.1')
    # parse command line arguments
    artist_name = 'maestro'

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--train_data_path', default='datasets/%s_64_conditioned_new_train.npy'%artist_name, type=str)
    parser.add_argument('--test_data_path', default='datasets/%s_64_conditioned_new_test.npy'%artist_name, type=str)
    parser.add_argument('--output_path', default='outputs/%s_64'%artist_name, type=str)   
    parser.add_argument('--model_name', default='VAE')
    parser.add_argument('--num_particles', default=4, type=int, help='number of particles')
    parser.add_argument('-n', '--num-epochs', default=301, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=3.0e-6, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help='whether to use cuda')
    parser.add_argument('-i-tsne', '--tsne_iter', default=100, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()
    args.output_path += '_%s_num_particle_%d/' % (args.model_name, args.num_particles)

    model = main(artist_name, args)
    