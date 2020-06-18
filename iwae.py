# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch
import math
import pyro
import pyro.optim
import pyro.poutine as poutine
from pyro.infer.abstract_infer import TracePosterior
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item

import warnings
from abc import ABCMeta, abstractmethod

from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_site_shape

import pyro
import pyro.ops.jit
from pyro.distributions.util import is_identically_zero, log_sum_exp
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import MultiFrameTensor, get_plate_stacks, is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan

def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_plate_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r

class Trace_IWAE(ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide. The gradient estimator includes
    partial Rao-Blackwellization for reducing the variance of the estimator when
    non-reparameterizable random variables are present. The Rao-Blackwellization is
    partial in that it only uses conditional independence information that is marked
    by :class:`~pyro.plate` contexts. For more fine-grained Rao-Blackwellization,
    see :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`.

    References

    [1] Automated Variational Inference in Probabilistic Programming,
        David Wingate, Theo Weber

    [2] Black Box Variational Inference,
        Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo_particles = []
        is_vectorized = self.vectorize_particles and self.num_particles > 1
        # particle_elbo_sum = 0.0

        # elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(guide_trace.log_prob_sum())
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = 0.0

            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    if is_vectorized:
                        log_prob_sum = site["log_prob"].detach().reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = torch_item(site["log_prob_sum"])

                    elbo_particle = elbo_particle + log_prob_sum

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    if is_vectorized:
                        log_prob_sum = log_prob.detach().reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = torch_item(site["log_prob_sum"])

                    elbo_particle = elbo_particle - log_prob_sum

            elbo_particles.append(elbo_particle)
        
        if is_vectorized:
            elbo_particles = elbo_particles[0]
        else:
            elbo_particles = torch.tensor(elbo_particles)  # no need to use .new*() here

        log_mean_weight = log_sum_exp(elbo_particles, dim=0) - math.log(self.num_particles)
        elbo = log_mean_weight.sum().item()

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss    

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float
        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        elbo_particles = []
        surrogate_elbo_particles = []
        is_vectorized = self.vectorize_particles and self.num_particles > 1
        tensor_holder = None

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0

            # compute elbo and surrogate elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    if is_vectorized:
                        log_prob_sum = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = site["log_prob_sum"]
                    elbo_particle = elbo_particle + log_prob_sum.detach()
                    surrogate_elbo_particle = surrogate_elbo_particle + log_prob_sum

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    if is_vectorized:
                        log_prob_sum = log_prob.reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = site["log_prob_sum"]

                    elbo_particle = elbo_particle - log_prob_sum.detach()

                    if not is_identically_zero(entropy_term):
                        surrogate_elbo_particle = surrogate_elbo_particle - log_prob_sum

                        if not is_identically_zero(score_function_term):
                            # link to the issue: https://github.com/uber/pyro/issues/1222
                            raise NotImplementedError

                    # if not is_identically_zero(score_function_term):
                    #     surrogate_elbo_particle = surrogate_elbo_particle

            if is_identically_zero(elbo_particle):
                if tensor_holder is not None:
                    elbo_particle = tensor_holder.new_zeros(tensor_holder.shape)
                    surrogate_elbo_particle = tensor_holder.new_zeros(tensor_holder.shape)
            else:  # elbo_particle is not None
                if tensor_holder is None:
                    tensor_holder = elbo_particle.new_empty(elbo_particle.shape)
                    # change types of previous `elbo_particle`s
                    for i in range(len(elbo_particles)):
                        elbo_particles[i] = tensor_holder.new_zeros(tensor_holder.shape)
                        surrogate_elbo_particles[i] = tensor_holder.new_zeros(tensor_holder.shape)

            elbo_particles.append(elbo_particle)
            surrogate_elbo_particles.append(surrogate_elbo_particle)

        if tensor_holder is None:
            return 0.

        if is_vectorized:
            elbo_particles = elbo_particles[0]
            surrogate_elbo_particles = surrogate_elbo_particles[0]
        else:
            elbo_particles = torch.stack(elbo_particles)
            surrogate_elbo_particles = torch.stack(surrogate_elbo_particles)

        log_weights = elbo_particles
        log_mean_weight = log_sum_exp(elbo_particles, dim=0) - math.log(self.num_particles)
        elbo = log_mean_weight.sum().item()

        # collect parameters to train from model and guide
        trainable_params = any(site["type"] == "param"
                               for trace in (model_trace, guide_trace)
                               for site in trace.nodes.values())

        if trainable_params and getattr(surrogate_elbo_particles, 'requires_grad', False):
            normalized_weights = (log_weights - log_mean_weight).exp()
            surrogate_elbo = (normalized_weights * surrogate_elbo_particles).sum() / self.num_particles
            surrogate_loss = -surrogate_elbo
            surrogate_loss.backward()

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

