# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch

import pyro
import pyro.optim
import pyro.poutine as poutine
from pyro.infer.abstract_infer import TracePosterior
# from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item

import warnings
from abc import ABCMeta, abstractmethod

from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_site_shape

import pyro
import pyro.ops.jit
from pyro.distributions.util import is_identically_zero
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import MultiFrameTensor, get_plate_stacks, is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan


class IWAE_BOUND(object, metaclass=ABCMeta):
    """
    :class:`IWAE_BOUND` is the top-level interface for stochastic variational
    inference via optimization of the evidence lower bound.

    Most users will not interact with this base class :class:`ELBO` directly;
    instead they will create instances of derived classes:
    :class:`~pyro.infer.trace_elbo.Trace_ELBO`,
    :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`, or
    :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO`.

    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is only required when enumerating
        over sample sites in parallel, e.g. if a site sets
        ``infer={"enumerate": "parallel"}``. If omitted, ELBO may guess a valid
        value by running the (model,guide) pair once, however this guess may
        be incorrect if model or guide structure is dynamic.
    :param bool vectorize_particles: Whether to vectorize the ELBO computation
        over `num_particles`. Defaults to False. This requires static structure
        in model and guide.
    :param bool strict_enumeration_warning: Whether to warn about possible
        misuse of enumeration, i.e. that
        :class:`pyro.infer.traceenum_elbo.TraceEnum_ELBO` is used iff there
        are enumerated sample sites.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer. When this is True, all :class:`torch.jit.TracerWarning` will
        be ignored. Defaults to False.
    :param bool jit_options: Optional dict of options to pass to
        :func:`torch.jit.trace` , e.g. ``{"check_trace": True}``.
    :param bool retain_graph: Whether to retain autograd graph during an SVI
        step. Defaults to None (False).
    :param float tail_adaptive_beta: Exponent beta with ``-1.0 <= beta < 0.0`` for
        use with `TraceTailAdaptive_ELBO`.

    References

    [1] `Automated Variational Inference in Probabilistic Programming`
    David Wingate, Theo Weber

    [2] `Black Box Variational Inference`,
    Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def __init__(self,
                 num_particles=1,
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=False,
                 strict_enumeration_warning=True,
                 ignore_jit_warnings=False,
                 jit_options=None,
                 retain_graph=None,
                 tail_adaptive_beta=-1.0):
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting
        self.max_plate_nesting = max_plate_nesting
        self.num_particles = num_particles
        self.vectorize_particles = vectorize_particles
        self.retain_graph = retain_graph
        if self.vectorize_particles and self.num_particles > 1:
            self.max_plate_nesting += 1
        self.strict_enumeration_warning = strict_enumeration_warning
        self.ignore_jit_warnings = ignore_jit_warnings
        self.jit_options = jit_options
        self.tail_adaptive_beta = tail_adaptive_beta

    def _guess_max_plate_nesting(self, model, guide, args, kwargs):
        """
        Guesses max_plate_nesting by running the (model,guide) pair once
        without enumeration. This optimistically assumes static model
        structure.
        """
        # Ignore validation to allow model-enumerated sites absent from the guide.
        with poutine.block():
            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        guide_trace = prune_subsample_sites(guide_trace)
        model_trace = prune_subsample_sites(model_trace)
        sites = [site
                 for trace in (model_trace, guide_trace)
                 for site in trace.nodes.values()
                 if site["type"] == "sample"]

        # Validate shapes now, since shape constraints will be weaker once
        # max_plate_nesting is changed from float('inf') to some finite value.
        # Here we know the traces are not enumerated, but later we'll need to
        # allow broadcasting of dims to the left of max_plate_nesting.
        if is_validation_enabled():
            guide_trace.compute_log_prob()
            model_trace.compute_log_prob()
            for site in sites:
                check_site_shape(site, max_plate_nesting=float('inf'))

        dims = [frame.dim
                for site in sites
                for frame in site["cond_indep_stack"]
                if frame.vectorized]
        self.max_plate_nesting = -min(dims) if dims else 0
        if self.vectorize_particles and self.num_particles > 1:
            self.max_plate_nesting += 1
        print('Guessed max_plate_nesting = {}'.format(self.max_plate_nesting))

    def _vectorized_num_particles(self, fn):
        """
        Wraps a callable inside an outermost :class:`~pyro.plate` to parallelize
        ELBO computation over `num_particles`, and to broadcast batch shapes of
        sample site functions in accordance with the `~pyro.plate` contexts
        within which they are embedded.

        :param fn: arbitrary callable containing Pyro primitives.
        :return: wrapped callable.
        """

        def wrapped_fn(*args, **kwargs):
            if self.num_particles == 1:
                return fn(*args, **kwargs)
            with pyro.plate("num_particles_vectorized", self.num_particles, dim=-self.max_plate_nesting):
                return fn(*args, **kwargs)

        return wrapped_fn

    def _get_vectorized_trace(self, model, guide, args, kwargs):
        """
        Wraps the model and guide to vectorize ELBO computation over
        ``num_particles``, and returns a single trace from the wrapped model
        and guide.
        """
        return self._get_trace(self._vectorized_num_particles(model),
                               self._vectorized_num_particles(guide),
                               args, kwargs)

    @abstractmethod
    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        raise NotImplementedError

    def _get_traces(self, model, guide, args, kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.vectorize_particles:
            if self.max_plate_nesting == float('inf'):
                self._guess_max_plate_nesting(model, guide, args, kwargs)
            yield self._get_vectorized_trace(model, guide, args, kwargs)
        else:
            for i in range(self.num_particles):
                yield self._get_trace(model, guide, args, kwargs)


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


class Trace_IWAE(IWAE_BOUND):
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
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(guide_trace.log_prob_sum())
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None

        # compute elbo and surrogate elbo
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle = elbo_particle + torch_item(site["log_prob_sum"])
                surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = surrogate_elbo_particle - entropy_term.sum()

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = surrogate_elbo_particle + (site * score_function_term).sum()

        return -elbo_particle, -surrogate_elbo_particle

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters
        """
        loss = 0.
        surrogate_loss = 0.
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(model_trace, guide_trace)
            surrogate_loss += surrogate_loss_particle / self.num_particles
            loss += loss_particle / self.num_particles
        warn_if_nan(surrogate_loss, "loss")
        return loss + (surrogate_loss - torch_item(surrogate_loss))

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(model_trace, guide_trace)
            loss += loss_particle / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = any(site["type"] == "param"
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values())

            if trainable_params and getattr(surrogate_loss_particle, 'requires_grad', False):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
        warn_if_nan(loss, "loss")
        return loss

class SVI(TracePosterior):
    """
    :param model: the model (callable containing Pyro primitives)
    :param guide: the guide (callable containing Pyro primitives)
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: ~pyro.optim.optim.PyroOptim
    :param loss: an instance of a subclass of :class:`~pyro.infer.elbo.ELBO`.
        Pyro provides three built-in losses:
        :class:`~pyro.infer.trace_elbo.Trace_ELBO`,
        :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`, and
        :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO`.
        See the :class:`~pyro.infer.elbo.ELBO` docs to learn how to implement
        a custom loss.
    :type loss: pyro.infer.elbo.ELBO
    :param num_samples: (DEPRECATED) the number of samples for Monte Carlo posterior approximation
    :param num_steps: (DEPRECATED) the number of optimization steps to take in ``run()``

    A unified interface for stochastic variational inference in Pyro. The most
    commonly used loss is ``loss=Trace_ELBO()``. See the tutorial
    `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for a discussion.
    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 loss,
                 loss_and_grads=None,
                 num_samples=0,
                 num_steps=0,
                 **kwargs):
        if num_steps:
            warnings.warn('The `num_steps` argument to SVI is deprecated and will be removed in '
                          'a future release. Use `SVI.step` directly to control the '
                          'number of iterations.', FutureWarning)
        if num_samples:
            warnings.warn('The `num_samples` argument to SVI is deprecated and will be removed in '
                          'a future release. Use `pyro.infer.Predictive` class to draw '
                          'samples from the posterior.', FutureWarning)

        self.model = model
        self.guide = guide
        self.optim = optim
        self.num_steps = num_steps
        self.num_samples = num_samples
        super().__init__(**kwargs)

        if not isinstance(optim, pyro.optim.PyroOptim):
            raise ValueError("Optimizer should be an instance of pyro.optim.PyroOptim class.")

        if isinstance(loss, IWAE_BOUND):
            self.loss = loss.loss
            self.loss_and_grads = loss.loss_and_grads
        else:
            if loss_and_grads is None:
                def _loss_and_grads(*args, **kwargs):
                    loss_val = loss(*args, **kwargs)
                    if getattr(loss_val, 'requires_grad', False):
                        loss_val.backward(retain_graph=True)
                    return loss_val
                loss_and_grads = _loss_and_grads
            self.loss = loss
            self.loss_and_grads = loss_and_grads

    def run(self, *args, **kwargs):
        """
        .. warning::
            This method is deprecated, and will be removed in a future release.
            For inference, use :meth:`step` directly, and for predictions,
            use the :class:`~pyro.infer.predictive.Predictive` class.
        """
        warnings.warn('The `SVI.run` method is deprecated and will be removed in a '
                      'future release. For inference, use `SVI.step` directly, '
                      'and for predictions, use the `pyro.infer.Predictive` class.',
                      FutureWarning)
        if self.num_steps > 0:
            with poutine.block():
                for i in range(self.num_steps):
                    self.step(*args, **kwargs)
        return super().run(*args, **kwargs)

    def _traces(self, *args, **kwargs):
        for i in range(self.num_samples):
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(self.model, trace=guide_trace)).get_trace(*args, **kwargs)
            yield model_trace, 1.0

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        with torch.no_grad():
            loss = self.loss(self.model, self.guide, *args, **kwargs)
            if isinstance(loss, tuple):
                # Support losses that return a tuple, e.g. ReweightedWakeSleep.
                return type(loss)(map(torch_item, loss))
            else:
                return torch_item(loss)

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        if isinstance(loss, tuple):
            # Support losses that return a tuple, e.g. ReweightedWakeSleep.
            return type(loss)(map(torch_item, loss))
        else:
            return torch_item(loss)

