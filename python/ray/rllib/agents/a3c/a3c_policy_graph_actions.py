"""Note: Keep in sync with changes to VTracePolicyGraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import gym

import ray
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.annotations import override


def agent_name_to_idx(name, self_id):
    agent_num = int(name[6])
    self_num = int(self_id[6])
    if agent_num > self_num:
        return agent_num - 1
    else:
        return agent_num


class A3CLoss(object):
    def __init__(self,
                 action_dist,
                 actions,
                 advantages,
                 v_target,
                 vf,
                 vf_loss_coeff=0.5,
                 entropy_coeff=-0.01):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff +
                           self.entropy * entropy_coeff)


class A3CPolicyGraph(LearningRateSchedule, TFPolicyGraph):
    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.a3c.a3c.DEFAULT_CONFIG, **config)
        self.config = config
        self.sess = tf.get_default_session()

        # Extract info from config
        self.num_other_agents = config['num_other_agents']
        self.agent_id = config['agent_id']

        # Setup the policy
        self.observations = tf.placeholder(
            tf.float32, [None] + list(observation_space.shape))

        # Add other agents actions placeholder
        self.others_actions = tf.placeholder(tf.int32, 
            shape=(None, self.num_other_agents), name="others_actions")

        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])
        prev_actions = ModelCatalog.get_action_placeholder(action_space)
        prev_rewards = tf.placeholder(tf.float32, [None], name="prev_reward")
        self.model = ModelCatalog.get_model({
            "obs": self.observations,
            "others_actions": self.others_actions,
            "prev_actions": prev_actions,
            "prev_rewards": prev_rewards,
            "is_training": self._get_is_training_placeholder(),
        }, observation_space, logit_dim, self.config["model"])
        action_dist = dist_class(self.model.outputs)
        self.vf = self.model.value_function()
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)

        # Setup the policy loss
        if isinstance(action_space, gym.spaces.Box):
            ac_size = action_space.shape[0]
            actions = tf.placeholder(tf.float32, [None, ac_size], name="ac")
        elif isinstance(action_space, gym.spaces.Discrete):
            actions = tf.placeholder(tf.int64, [None], name="ac")
        else:
            raise UnsupportedSpaceException(
                "Action space {} is not supported for A3C.".format(
                    action_space))
        advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.v_target = tf.placeholder(tf.float32, [None], name="v_target")
        self.loss = A3CLoss(action_dist, actions, advantages, self.v_target,
                            self.vf, self.config["vf_loss_coeff"],
                            self.config["entropy_coeff"])

        # Initialize TFPolicyGraph
        loss_in = [
            ("obs", self.observations),
            ("others_actions", self.others_actions),
            ("actions", actions),
            ("prev_actions", prev_actions),
            ("prev_rewards", prev_rewards),
            ("advantages", advantages),
            ("value_targets", self.v_target),
        ]
        LearningRateSchedule.__init__(self, self.config["lr"],
                                      self.config["lr_schedule"])
        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=self.observations,
            action_sampler=action_dist.sample(),
            action_prob=action_dist.sampled_action_prob(),
            loss=self.loss.total_loss,
            model=self.model,
            loss_inputs=loss_in,
            state_inputs=self.model.state_in,
            state_outputs=self.model.state_out,
            prev_action_input=prev_actions,
            prev_reward_input=prev_rewards,
            seq_lens=self.model.seq_lens,
            max_seq_len=self.config["model"]["max_seq_len"])

        self.stats_fetches = {
            "stats": {
                "cur_lr": tf.cast(self.cur_lr, tf.float64),
                "policy_loss": self.loss.pi_loss,
                "policy_entropy": self.loss.entropy,
                "grad_gnorm": tf.global_norm(self._grads),
                "var_gnorm": tf.global_norm(self.var_list),
                "vf_loss": self.loss.vf_loss,
                "vf_explained_var": explained_variance(self.v_target, self.vf),
            },
        }

        self.sess.run(tf.global_variables_initializer())

    @override(PolicyGraph)
    def get_initial_state(self):
        return self.model.state_init

    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        # Extract and storeother agents' actions.
        others_actions = self.extract_last_actions_from_episodes(
            other_agent_batches, batch_type=True)
        sample_batch['others_actions'] = others_actions

        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(len(self.model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            prev_action = sample_batch['prev_actions'][-1]
            prev_reward = sample_batch['prev_rewards'][-1]
            
            last_r = self._value(sample_batch["new_obs"][-1], 
                                 others_actions[-1], prev_action, prev_reward, 
                                 *next_state)

        return compute_advantages(sample_batch, last_r, self.config["gamma"],
                                  self.config["lambda"])

    def extract_last_actions_from_episodes(self, episodes, batch_type=False,
                                           exclude_self=True):
        """Pulls every other agent's previous actions out of structured data.
        Args:
            episodes: the structured data type. Typically a dict of episode
                objects.
            batch_type: if True, the structured data is a dict of tuples,
                where the second tuple element is the relevant dict containing
                previous actions.
            exclude_self: If True, will not include the agent's own actions in
                the returned array.
        Returns: a real valued array of size [batch, num_other_agents] (meaning
            each agents' actions goes down one column, each row is a timestep)
        """
        if episodes is None:
            print("Why are there no episodes?")
            import pdb; pdb.set_trace()

        if type(episodes) == dict and 'all_agents_actions' in episodes.keys():
            if exclude_self:
                self_index = agent_name_to_idx(self.agent_id)
                others_actions = [e for i, e in enumerate(
                    episodes['all_agents_actions']) if self_index != i]
                return np.reshape(np.array(others_actions, [1,-1]))
            else:
                return np.reshape(
                    np.array(episodes['all_agents_actions']), [1,-1])
        
        # Need to sort agent IDs so same agent is consistently in
        # same part of input space.
        agent_ids = sorted(episodes.keys())
        prev_actions = []

        for agent_id in agent_ids:
            if exclude_self and agent_id == self.agent_id:
                continue
            if batch_type:
                prev_actions.append(episodes[agent_id][1]['actions'])
            else:
                prev_actions.append(
                    [e.prev_action for e in episodes[agent_id]])

        # Need a transpose to make a [batch_size, num_other_agents] tensor
        return np.transpose(np.array(prev_actions))

    @override(TFPolicyGraph)
    def _build_compute_actions(self,
                               builder,
                               obs_batch,
                               state_batches=None,
                               prev_action_batch=None,
                               prev_reward_batch=None,
                               episodes=None):
        state_batches = state_batches or []
        if len(self._state_inputs) != len(state_batches):
            raise ValueError(
                "Must pass in RNN state batches for placeholders {}, got {}".
                format(self._state_inputs, state_batches))
        builder.add_feed_dict(self.extra_compute_action_feed_dict())

        # Need to compute other agents' actions.
        others_actions = self.extract_last_actions_from_episodes(episodes)

        # Debugging:
        # if self.agent_id == 'agent-0':
            # agent_ids = sorted(episodes.keys())
            # prev_actions = []
            # for agent_id in agent_ids:
            #     prev_actions.append([e.prev_action for e in episodes[agent_id]])
            # print(np.transpose(np.array(prev_actions)))

        builder.add_feed_dict({self._obs_input: obs_batch,
                               self.others_actions: others_actions})

        if state_batches:
            builder.add_feed_dict({self._seq_lens: np.ones(len(obs_batch))})
        if self._prev_action_input is not None and prev_action_batch:
            builder.add_feed_dict({self._prev_action_input: prev_action_batch})
        if self._prev_reward_input is not None and prev_reward_batch:
            builder.add_feed_dict({self._prev_reward_input: prev_reward_batch})
        builder.add_feed_dict({self._is_training: False})
        builder.add_feed_dict(dict(zip(self._state_inputs, state_batches)))
        fetches = builder.add_fetches([self._sampler] + self._state_outputs +
                                      [self.extra_compute_action_fetches()])
        return fetches[0], fetches[1:-1], fetches[-1]


    @override(TFPolicyGraph)
    def gradients(self, optimizer):
        grads = tf.gradients(self._loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
        clipped_grads = list(zip(self.grads, self.var_list))
        return clipped_grads

    @override(TFPolicyGraph)
    def extra_compute_grad_fetches(self):
        return self.stats_fetches

    @override(TFPolicyGraph)
    def extra_compute_action_fetches(self):
        return dict(
            TFPolicyGraph.extra_compute_action_fetches(self),
            **{"vf_preds": self.vf})

    def _value(self, ob, others_actions, prev_action, prev_reward, *args):
        feed_dict = {self.observations: [ob], 
                     self.others_actions: [others_actions], 
                     self.model.seq_lens: [1],
                     self._prev_action_input: [prev_action],
                     self._prev_reward_input: [prev_reward]}
        assert len(args) == len(self.model.state_in), \
            (args, self.model.state_in)
        for k, v in zip(self.model.state_in, args):
            feed_dict[k] = v
        vf = self.sess.run(self.vf, feed_dict)
        return vf[0]
