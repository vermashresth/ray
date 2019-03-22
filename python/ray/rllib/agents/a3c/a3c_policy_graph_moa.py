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

class MOALoss(object):
    def __init__(self, action_logits, true_actions, num_actions, num_agents,
                 loss_weight=1.0):
        """Train MOA model with supervised cross entropy loss on a trajectory.

        The model is trying to predict others' actions at timestep t+1 given all 
        actions at timestep t.

        Returns:
            A scalar loss tensor (cross-entropy loss).
        """
        # Reshape to [B, N, A]
        action_logits = tf.reshape(action_logits, [-1, num_agents, num_actions])

        # Remove the prediction for the final step, since t+1 is not known for
        # this step.
        action_logits = action_logits[:-1, :, :]  # [B, N, A]

        # Ensure ground truth others' actions to start at the t+1 step, since 
        # we want to predict the actions at t+1 from t.
        true_actions = true_actions[1:, :]  # [B, N]

        # Compute softmax cross entropy
        flat_logits = tf.reshape(action_logits, [-1, num_actions])
        flat_labels = tf.reshape(true_actions, [-1])
        self.ce_per_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=flat_labels, logits=flat_logits)
        self.total_loss = tf.reduce_mean(self.ce_per_entry)
        tf.Print(self.total_loss, [self.total_loss], message="MOA CE loss")
        
        #TODO(natashajaques): Add something to train only when other agents are
        # visible here


class A3CPolicyGraph(LearningRateSchedule, TFPolicyGraph):
    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.a3c.a3c.DEFAULT_CONFIG, **config)
        self.config = config
        self.sess = tf.get_default_session()

        # Extract info from config
        self.num_other_agents = config['num_other_agents']
        self.agent_id = config['agent_id']
        self.moa_weight = config['model']['custom_options']['moa_weight']

        # Setup the policy
        self.observations = tf.placeholder(
            tf.float32, [None] + list(observation_space.shape))

        # Add other agents actions placeholder for MOA preds
        # Add 1 to include own action. Note: agent's own actions will always 
        # form the first column of this tensor.
        self.others_actions = tf.placeholder(tf.int32, 
            shape=(None, self.num_other_agents + 1), name="others_actions")

        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])
        prev_actions = ModelCatalog.get_action_placeholder(action_space)
        prev_rewards = tf.placeholder(tf.float32, [None], name="prev_reward")

        # Compute output size of model of other agents (MOA)
        moa_dim = logit_dim * (self.num_other_agents + 1)
        
        # We now create two models, one for the policy, and one for the model
        # of other agents (MOA)
        self.rl_model, self.moa = ModelCatalog.get_double_lstm_model({
                "obs": self.observations,
                "others_actions": self.others_actions,
                "prev_actions": prev_actions,
                "prev_rewards": prev_rewards,
                "is_training": self._get_is_training_placeholder(),
            }, observation_space, logit_dim, moa_dim, self.config["model"], 
            lstm1_name="policy", lstm2_name="moa")
        
        action_dist = dist_class(self.rl_model.outputs)
        self.vf = self.rl_model.value_function()
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
        self.rl_loss = A3CLoss(action_dist, actions, advantages, self.v_target,
                            self.vf, self.config["vf_loss_coeff"],
                            self.config["entropy_coeff"])

        # Setup the MOA loss
        self.moa_preds = self.moa.outputs
        self.moa_loss = MOALoss(self.moa_preds, self.others_actions, 
                                logit_dim, self.num_other_agents + 1,
                                loss_weight=self.moa_weight)
        self.moa_action_probs = tf.nn.softmax(self.moa_preds)

        # Total loss
        self.total_loss = self.rl_loss.total_loss + self.moa_loss.total_loss

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
            loss=self.total_loss,
            model=self.rl_model,
            loss_inputs=loss_in,
            state_inputs=self.rl_model.state_in + self.moa.state_in,
            state_outputs=self.rl_model.state_out + self.moa.state_out,
            prev_action_input=prev_actions,
            prev_reward_input=prev_rewards,
            seq_lens=self.rl_model.seq_lens,
            max_seq_len=self.config["model"]["max_seq_len"])

        self.stats = {
            "cur_lr": tf.cast(self.cur_lr, tf.float64),
            "policy_loss": self.rl_loss.pi_loss,
            "policy_entropy": self.rl_loss.entropy,
            "grad_gnorm": tf.global_norm(self._grads),
            "var_gnorm": tf.global_norm(self.var_list),
            "vf_loss": self.rl_loss.vf_loss,
            "vf_explained_var": explained_variance(self.v_target, self.vf),
            "moa_loss": self.moa_loss.total_loss
        }

        self.sess.run(tf.global_variables_initializer())

    @override(PolicyGraph)
    def get_initial_state(self):
        return self.rl_model.state_init + self.moa.state_init

    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        # Extract matrix of self and other agents' actions.
        own_actions = np.atleast_2d(np.array(sample_batch['actions']))
        own_actions = np.reshape(own_actions, [-1,1])
        all_actions = self.extract_last_actions_from_episodes(
            other_agent_batches, own_actions=own_actions, batch_type=True)
        sample_batch['others_actions'] = all_actions

        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(len(self.rl_model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            prev_action = sample_batch['prev_actions'][-1]
            prev_reward = sample_batch['prev_rewards'][-1]
            
            last_r = self._value(sample_batch["new_obs"][-1], 
                                 all_actions[-1], prev_action, prev_reward, 
                                 *next_state)

        sample_batch = compute_advantages(sample_batch, last_r, self.config["gamma"],
                                          self.config["lambda"])
        return sample_batch

    def extract_last_actions_from_episodes(self, episodes, batch_type=False, 
                                           own_actions=None):
        """Pulls every other agent's previous actions out of structured data.
        Args:
            episodes: the structured data type. Typically a dict of episode
                objects.
            batch_type: if True, the structured data is a dict of tuples,
                where the second tuple element is the relevant dict containing
                previous actions.
            own_actions: an array of the agents own actions. If provided, will
                be the first column of the created action matrix.
        Returns: a real valued array of size [batch, num_other_agents] (meaning
            each agents' actions goes down one column, each row is a timestep)
        """
        if episodes is None:
            print("Why are there no episodes?")
            import pdb; pdb.set_trace()

        # Need to sort agent IDs so same agent is consistently in
        # same part of input space.
        agent_ids = sorted(episodes.keys())
        prev_actions = []

        for agent_id in agent_ids:
            if agent_id == self.agent_id:
                continue
            if batch_type:
                prev_actions.append(episodes[agent_id][1]['actions'])
            else:
                prev_actions.append(
                    [e.prev_action for e in episodes[agent_id]])
        
        # Need a transpose to make a [batch_size, num_other_agents] tensor
        all_actions = np.transpose(np.array(prev_actions))
        
        # Attach agents own actions as column 1
        if own_actions is not None:
            all_actions = np.hstack((own_actions, all_actions))

        return all_actions

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

        # Extract matrix of other agents' past actions, including agent's own
        own_actions = np.atleast_2d(np.array(
            [e.prev_action for e in episodes[self.agent_id]]))
        all_actions = self.extract_last_actions_from_episodes(
            episodes, own_actions=own_actions)

        # Debugging:
        # if self.agent_id == 'agent-0':
            # agent_ids = sorted(episodes.keys())
            # prev_actions = []
            # for agent_id in agent_ids:
            #     prev_actions.append([e.prev_action for e in episodes[agent_id]])
            # print(np.transpose(np.array(prev_actions)))

        builder.add_feed_dict({self._obs_input: obs_batch,
                               self.others_actions: all_actions})

        if state_batches:
            seq_lens = np.ones(len(obs_batch))
            builder.add_feed_dict({self._seq_lens: seq_lens,
                                   self.moa.seq_lens: seq_lens})
        if self._prev_action_input is not None and prev_action_batch:
            builder.add_feed_dict({self._prev_action_input: prev_action_batch})
        if self._prev_reward_input is not None and prev_reward_batch:
            builder.add_feed_dict({self._prev_reward_input: prev_reward_batch})
        builder.add_feed_dict({self._is_training: False})
        builder.add_feed_dict(dict(zip(self._state_inputs, state_batches)))
        fetches = builder.add_fetches([self._sampler] + self._state_outputs +
                                      [self.extra_compute_action_fetches()])
        return fetches[0], fetches[1:-1], fetches[-1]

    def _get_loss_inputs_dict(self, batch):
        # Override parent function to add seq_lens to tensor for additional LSTM
        loss_inputs = super(A3CPolicyGraph, self)._get_loss_inputs_dict(batch)
        loss_inputs[self.moa.seq_lens] = loss_inputs[self._seq_lens]
        return loss_inputs

    @override(TFPolicyGraph)
    def gradients(self, optimizer):
        grads = tf.gradients(self._loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
        clipped_grads = list(zip(self.grads, self.var_list))
        return clipped_grads

    @override(TFPolicyGraph)
    def extra_compute_grad_fetches(self):
        return {
            "stats": self.stats,
        }

    @override(TFPolicyGraph)
    def extra_compute_action_fetches(self):
        return dict(
            TFPolicyGraph.extra_compute_action_fetches(self),
            **{"vf_preds": self.vf})

    def _value(self, ob, others_actions, prev_action, prev_reward, *args):
        feed_dict = {self.observations: [ob], 
                     self.others_actions: [others_actions], 
                     self.rl_model.seq_lens: [1],
                     self._prev_action_input: [prev_action],
                     self._prev_reward_input: [prev_reward]}
        assert len(args) == len(self.rl_model.state_in), \
            (args, self.rl_model.state_in)
        for k, v in zip(self.rl_model.state_in, args):
            feed_dict[k] = v
        vf = self.sess.run(self.vf, feed_dict)
        return vf[0]
