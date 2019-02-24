#!/usr/bin/env python
"""Example of using PBT with RLlib.

Note that this requires a cluster with at least 8 GPUs in order for all trials
to run concurrently, otherwise PBT will round-robin train the trials which
is less efficient (or you can set {"gpu": 0} to use CPUs for SGD instead).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import ray
from ray.tune import run_experiments, sample_from
from ray.tune.schedulers import PopulationBasedTraining

if __name__ == "__main__":

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        reward_attr="episode_reward_mean",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        },
        custom_explore_fn=explore)

    ray.init()
    run_experiments(
        {
            "pbt_humanoid_test": {
                "run": "PPO",
                "env": "Humanoid-v1",
                "num_samples": 8,
                "config": {
                    "kl_coeff": 1.0,
                    "num_workers": 8,
                    "num_gpus": 1,
                    "model": {
                        "free_log_std": True
                    },
                    # These params are tuned from a fixed starting value.
                    "lambda": 0.95,
                    "clip_param": 0.2,
                    "lr": 1e-4,
                    # These params start off randomly drawn from a set.
                    "num_sgd_iter": sample_from(
                        lambda spec: random.choice([10, 20, 30])),
                    "sgd_minibatch_size": sample_from(
                        lambda spec: random.choice([128, 512, 2048])),
                    "train_batch_size": sample_from(
                        lambda spec: random.choice([10000, 20000, 40000]))
                },
            },
        },
        scheduler=pbt)
