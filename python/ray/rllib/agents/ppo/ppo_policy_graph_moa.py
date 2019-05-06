from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

import ray
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.explained_variance import explained_variance

logger = logging.getLogger(__name__)

class MOALoss(object):
    def __init__(self, action_logits, true_actions, num_actions,
                 loss_weight=1.0, others_visibility=None):
        """Train MOA model with supervised cross entropy loss on a trajectory.

        The model is trying to predict others' actions at timestep t+1 given all
        actions at timestep t.

        Returns:
            A scalar loss tensor (cross-entropy loss).
        """
        # Remove the prediction for the final step, since t+1 is not known for
        # this step.
        action_logits = action_logits[:-1, :, :]  # [B, N, A]

        # Remove first agent (self) and first action, because we want to predict
        # the t+1 actions from other agents actions at t.
        true_actions = true_actions[1:, 1:]  # [B, N]

        # Compute softmax cross entropy
        flat_logits = tf.reshape(action_logits, [-1, num_actions])
        flat_labels = tf.reshape(true_actions, [-1])
        self.ce_per_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=flat_labels, logits=flat_logits)

        # Zero out the loss if the other agent isn't visible to this one.
        if others_visibility is not None:
            # Remove first entry in ground truth visibility and flatten
            others_visibility = tf.reshape(others_visibility[1:,:], [-1])
            self.ce_per_entry *= tf.cast(others_visibility, tf.float32)

        self.total_loss = tf.reduce_mean(self.ce_per_entry)
        tf.Print(self.total_loss, [self.total_loss], message="MOA CE loss")

class PPOLoss(object):
    def __init__(self,
                 action_space,
                 value_targets,
                 advantages,
                 actions,
                 logits,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            action_space: Environment observation space specification.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        dist_cls, _ = ModelCatalog.get_action_dist(action_space, {})
        prev_dist = dist_cls(logits)
        # Make loss functions.
        logp_ratio = tf.exp(
            curr_action_dist.logp(actions) - prev_dist.logp(actions))
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss


class PPOPolicyGraph(LearningRateSchedule, TFPolicyGraph):
    def __init__(self,
                 observation_space,
                 action_space,
                 config,
                 existing_inputs=None):
        """
        Arguments:
            observation_space: Environment observation space specification.
            action_space: Environment action space specification.
            config (dict): Configuration values for PPO graph.
            existing_inputs (list): Optional list of tuples that specify the
                placeholders upon which the graph should be built upon.
        """
        config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
        # Extract info from config
        self.num_other_agents = config['num_other_agents']
        self.agent_id = config['agent_id']
        self.moa_weight = config['model']['custom_options']['moa_weight']
        self.train_moa_only_when_visible = \
            config['model']['custom_options']['train_moa_only_when_visible']

        self.sess = tf.get_default_session()
        self.action_space = action_space
        self.config = config
        self.kl_coeff_val = self.config["kl_coeff"]
        self.kl_target = self.config["kl_target"]
        dist_cls, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])

        if existing_inputs:
            if not self.train_moa_only_when_visible:
                obs_ph, value_targets_ph, adv_ph, act_ph, \
                    logits_ph, vf_preds_ph, prev_actions_ph, prev_rewards_ph, \
                    others_action_ph = existing_inputs[:9]
                existing_state_in = existing_inputs[9:-1]
                existing_seq_lens = existing_inputs[-1]
            else:
                obs_ph, value_targets_ph, adv_ph, act_ph, \
                logits_ph, vf_preds_ph, prev_actions_ph, prev_rewards_ph, \
                others_action_ph, others_visibility_ph  = existing_inputs[:10]
                existing_state_in = existing_inputs[10:-1]
                existing_seq_lens = existing_inputs[-1]
        else:
            # Extract info from config
            self.num_other_agents = config['num_other_agents']
            self.agent_id = config['agent_id']
            obs_ph = tf.placeholder(
                tf.float32,
                name="obs",
                shape=(None, ) + observation_space.shape)
            adv_ph = tf.placeholder(
                tf.float32, name="advantages", shape=(None, ))
            act_ph = ModelCatalog.get_action_placeholder(action_space)
            logits_ph = tf.placeholder(
                tf.float32, name="logits", shape=(None, logit_dim))
            vf_preds_ph = tf.placeholder(
                tf.float32, name="vf_preds", shape=(None, ))
            value_targets_ph = tf.placeholder(
                tf.float32, name="value_targets", shape=(None, ))
            prev_actions_ph = ModelCatalog.get_action_placeholder(action_space)
            prev_rewards_ph = tf.placeholder(
                tf.float32, [None], name="prev_reward")
            # Add other agents actions placeholder
            others_action_ph = tf.placeholder(tf.int32,
                                                 shape=(None, self.num_other_agents),
                                                 name="others_actions")
            # 0/1 multiplier array representing whether each agent is visible to
            # the current agent.
            if self.train_moa_only_when_visible:
                others_visibility_ph = tf.placeholder(tf.int32,
                                                        shape=(None, self.num_other_agents), name="others_visibility")
            else:
                others_visibility_ph = None
            existing_state_in = None
            existing_seq_lens = None
        self.observations = obs_ph
        self.prev_actions = prev_actions_ph
        self.prev_rewards = prev_rewards_ph
        self.others_actions = others_action_ph
        self.others_visible = others_visibility_ph

        self.loss_in = [
            ("obs", obs_ph),
            ("value_targets", value_targets_ph),
            ("advantages", adv_ph),
            ("actions", act_ph),
            ("logits", logits_ph),
            ("vf_preds", vf_preds_ph),
            ("prev_actions", prev_actions_ph),
            ("prev_rewards", prev_rewards_ph),
            ("others_actions", others_action_ph),
        ]
        if self.train_moa_only_when_visible:
            self.loss_in.append(('others_visibility', self.others_visibility))

        self.model = ModelCatalog.get_model(
            {
                "obs": obs_ph,
                "others_actions": self.others_actions,
                "prev_actions": prev_actions_ph,
                "prev_rewards": prev_rewards_ph,
                "is_training": self._get_is_training_placeholder(),
            },
            observation_space,
            logit_dim,
            self.config["model"],
            state_in=existing_state_in,
            seq_lens=existing_seq_lens)
        # Compute output size of model of other agents (MOA)
        self.num_actions = logit_dim * self.num_other_agents
        self.moa_dim = logit_dim * self.num_other_agents
        self.moa = self.model.moa_model(self.moa_dim)

        # KL Coefficient
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)

        self.logits = self.model.outputs
        curr_action_dist = dist_cls(self.logits)
        self.sampler = curr_action_dist.sample()
        if self.config["use_gae"]:
            if self.config["vf_share_layers"]:
                self.value_function = self.model.value_function()
            else:
                vf_config = self.config["model"].copy()
                # Do not split the last layer of the value function into
                # mean parameters and standard deviation parameters and
                # do not make the standard deviations free variables.
                vf_config["free_log_std"] = False
                if vf_config["use_lstm"]:
                    vf_config["use_lstm"] = False
                    logger.warning(
                        "It is not recommended to use a LSTM model with "
                        "vf_share_layers=False (consider setting it to True). "
                        "If you want to not share layers, you can implement "
                        "a custom LSTM model that overrides the "
                        "value_function() method.")
                with tf.variable_scope("value_function"):
                    self.value_function = ModelCatalog.get_model({
                        "obs": obs_ph,
                        "prev_actions": prev_actions_ph,
                        "prev_rewards": prev_rewards_ph,
                        "is_training": self._get_is_training_placeholder(),
                    }, observation_space, 1, vf_config).outputs
                    self.value_function = tf.reshape(self.value_function, [-1])
        else:
            self.value_function = tf.zeros(shape=tf.shape(obs_ph)[:1])

        if self.model.state_in:
            max_seq_len = tf.reduce_max(self.model.seq_lens)
            mask = tf.sequence_mask(self.model.seq_lens, max_seq_len)
            mask = tf.reshape(mask, [-1])
        else:
            mask = tf.ones_like(adv_ph, dtype=tf.bool)

        # Setup the MOA loss
        self.moa_preds = tf.reshape(  # Reshape to [B,N,A]
            self.moa.outputs, [-1, self.num_other_agents, self.num_actions])
        self.moa_loss = MOALoss(self.moa_preds, self.others_actions,
                                self.num_actions, loss_weight=self.moa_weight,
                                others_visibility=self.others_visibility)
        self.moa_action_probs = tf.nn.softmax(self.moa_preds)

        self.loss_obj = PPOLoss(
            action_space,
            value_targets_ph,
            adv_ph,
            act_ph,
            logits_ph,
            vf_preds_ph,
            curr_action_dist,
            self.value_function,
            self.kl_coeff,
            mask,
            entropy_coeff=self.config["entropy_coeff"],
            clip_param=self.config["clip_param"],
            vf_clip_param=self.config["vf_clip_param"],
            vf_loss_coeff=self.config["vf_loss_coeff"],
            use_gae=self.config["use_gae"])

        LearningRateSchedule.__init__(self, self.config["lr"],
                                      self.config["lr_schedule"])
        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=obs_ph,
            action_sampler=self.sampler,
            action_prob=curr_action_dist.sampled_action_prob(),
            loss=self.loss_obj.loss + self.moa_loss.total_loss,
            model=self.model,
            loss_inputs=self.loss_in,
            state_inputs=self.model.state_in, #TODO(@evinitsky) add moa in when using LSTM
            state_outputs=self.model.state_out,#TODO(@evinitsky) add moa out when using LSTM
            prev_action_input=prev_actions_ph,
            prev_reward_input=prev_rewards_ph,
            seq_lens=self.model.seq_lens,
            max_seq_len=config["model"]["max_seq_len"])

        self.sess.run(tf.global_variables_initializer())
        self.explained_variance = explained_variance(value_targets_ph,
                                                     self.value_function)
        self.stats_fetches = {
            "cur_kl_coeff": self.kl_coeff,
            "cur_lr": tf.cast(self.cur_lr, tf.float64),
            "total_loss": self.loss_obj.loss,
            "policy_loss": self.loss_obj.mean_policy_loss,
            "vf_loss": self.loss_obj.mean_vf_loss,
            "vf_explained_var": self.explained_variance,
            "kl": self.loss_obj.mean_kl,
            "entropy": self.loss_obj.mean_entropy,
            "moa_loss": self.moa_loss.total_loss
        }

    @override(TFPolicyGraph)
    def copy(self, existing_inputs):
        """Creates a copy of self using existing input placeholders."""
        return PPOPolicyGraph(
            self.observation_space,
            self.action_space,
            self.config,
            existing_inputs=existing_inputs)

    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        # Extract matrix of self and other agents' actions.
        own_actions = np.atleast_2d(np.array(sample_batch['actions']))
        own_actions = np.reshape(own_actions, [-1, 1])
        all_actions = self.extract_last_actions_from_episodes(
            other_agent_batches, own_actions=own_actions, batch_type=True)
        sample_batch['others_actions'] = all_actions

        if self.train_moa_only_when_visible:
            sample_batch['others_visibility'] = \
                self.get_agent_visibility_multiplier(sample_batch)

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

    def get_agent_visibility_multiplier(self, trajectory):
        traj_len = len(trajectory['infos'])
        visibility = np.zeros((traj_len, self.num_other_agents))
        vis_lists = [info['visible_agents'] for info in trajectory['infos']]
        for i, v in enumerate(vis_lists):
            vis_agents = [int(a) for a in v]
            visibility[i, vis_agents] = 1
        return visibility

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

    @override(TFPolicyGraph)
    def gradients(self, optimizer):
        if self.config["grad_clip"] is not None:
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              tf.get_variable_scope().name)
            grads = tf.gradients(self._loss, self.var_list)
            self.grads, _ = tf.clip_by_global_norm(grads,
                                                   self.config["grad_clip"])
            clipped_grads = list(zip(self.grads, self.var_list))
            return clipped_grads
        else:
            return optimizer.compute_gradients(
                self._loss, colocate_gradients_with_ops=True)

    @override(PolicyGraph)
    def get_initial_state(self):
        return self.model.state_init

    @override(TFPolicyGraph)
    def extra_compute_action_fetches(self):
        return dict(
            TFPolicyGraph.extra_compute_action_fetches(self), **{
                "vf_preds": self.value_function,
                "logits": self.logits
            })

    @override(TFPolicyGraph)
    def extra_compute_grad_fetches(self):
        return self.stats_fetches

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff_val *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff_val *= 0.5
        self.kl_coeff.load(self.kl_coeff_val, session=self.sess)
        return self.kl_coeff_val

    def _value(self, ob, others_actions, prev_action, prev_reward, *args):
        feed_dict = {
            self.observations: [ob],
            self.others_actions: [others_actions],
            self.prev_actions: [prev_action],
            self.prev_rewards: [prev_reward],
            self.model.seq_lens: [1]
        }
        assert len(args) == len(self.model.state_in), \
            (args, self.model.state_in)
        for k, v in zip(self.model.state_in, args):
            feed_dict[k] = v
        vf = self.sess.run(self.value_function, feed_dict)
        return vf[0]
