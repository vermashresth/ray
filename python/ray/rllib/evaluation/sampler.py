from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from collections import defaultdict, namedtuple
import logging
import numpy as np
import six.moves.queue as queue
import threading

from ray.rllib.evaluation.episode import MultiAgentEpisode, _flatten_action
from ray.rllib.evaluation.sample_batch_builder import \
    MultiAgentSampleBatchBuilder
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.atari_wrappers import get_wrapper_by_cls, MonitorEnv
from ray.rllib.models.action_dist import TupleActions
from ray.rllib.utils.tf_run_builder import TFRunBuilder

logger = logging.getLogger(__name__)
_large_batch_warned = False

RolloutMetrics = namedtuple(
    "RolloutMetrics",
    ["episode_length", "episode_reward", "agent_rewards", "custom_metrics"])

PolicyEvalData = namedtuple("PolicyEvalData", [
    "env_id", "agent_id", "obs", "info", "rnn_state", "prev_action",
    "prev_reward"
])


class SyncSampler(object):
    def __init__(self,
                 env,
                 policies,
                 policy_mapping_fn,
                 preprocessors,
                 obs_filters,
                 clip_rewards,
                 unroll_length,
                 callbacks,
                 horizon=None,
                 pack=False,
                 tf_sess=None,
                 clip_actions=True):
        self.base_env = BaseEnv.to_base_env(env)
        self.unroll_length = unroll_length
        self.horizon = horizon
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.preprocessors = preprocessors
        self.obs_filters = obs_filters
        self.extra_batches = queue.Queue()
        self.rollout_provider = _env_runner(
            self.base_env, self.extra_batches.put, self.policies,
            self.policy_mapping_fn, self.unroll_length, self.horizon,
            self.preprocessors, self.obs_filters, clip_rewards, clip_actions,
            pack, callbacks, tf_sess)
        self.metrics_queue = queue.Queue()

    def get_data(self):
        while True:
            item = next(self.rollout_provider)
            if isinstance(item, RolloutMetrics):
                self.metrics_queue.put(item)
            else:
                return item

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_extra_batches(self):
        extra = []
        while True:
            try:
                extra.append(self.extra_batches.get_nowait())
            except queue.Empty:
                break
        return extra


class AsyncSampler(threading.Thread):
    def __init__(self,
                 env,
                 policies,
                 policy_mapping_fn,
                 preprocessors,
                 obs_filters,
                 clip_rewards,
                 unroll_length,
                 callbacks,
                 horizon=None,
                 pack=False,
                 tf_sess=None,
                 clip_actions=True,
                 blackhole_outputs=False):
        for _, f in obs_filters.items():
            assert getattr(f, "is_concurrent", False), \
                "Observation Filter must support concurrent updates."
        self.base_env = BaseEnv.to_base_env(env)
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.extra_batches = queue.Queue()
        self.metrics_queue = queue.Queue()
        self.unroll_length = unroll_length
        self.horizon = horizon
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.preprocessors = preprocessors
        self.obs_filters = obs_filters
        self.clip_rewards = clip_rewards
        self.daemon = True
        self.pack = pack
        self.tf_sess = tf_sess
        self.callbacks = callbacks
        self.clip_actions = clip_actions
        self.blackhole_outputs = blackhole_outputs
        self.shutdown = False

    def run(self):
        try:
            self._run()
        except BaseException as e:
            self.queue.put(e)
            raise e

    def _run(self):
        if self.blackhole_outputs:
            queue_putter = (lambda x: None)
            extra_batches_putter = (lambda x: None)
        else:
            queue_putter = self.queue.put
            extra_batches_putter = (
                lambda x: self.extra_batches.put(x, timeout=600.0))
        rollout_provider = _env_runner(
            self.base_env, extra_batches_putter, self.policies,
            self.policy_mapping_fn, self.unroll_length, self.horizon,
            self.preprocessors, self.obs_filters, self.clip_rewards,
            self.clip_actions, self.pack, self.callbacks, self.tf_sess)
        while not self.shutdown:
            # The timeout variable exists because apparently, if one worker
            # dies, the other workers won't die with it, unless the timeout is
            # set to some large number. This is an empirical observation.
            item = next(rollout_provider)
            if isinstance(item, RolloutMetrics):
                self.metrics_queue.put(item)
            else:
                queue_putter(item)

    def get_data(self):
        rollout = self.queue.get(timeout=600.0)

        # Propagate errors
        if isinstance(rollout, BaseException):
            raise rollout

        return rollout

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_extra_batches(self):
        extra = []
        while True:
            try:
                extra.append(self.extra_batches.get_nowait())
            except queue.Empty:
                break
        return extra


def _env_runner(base_env,
                extra_batch_callback,
                policies,
                policy_mapping_fn,
                unroll_length,
                horizon,
                preprocessors,
                obs_filters,
                clip_rewards,
                clip_actions,
                pack,
                callbacks,
                tf_sess=None):
    """This implements the common experience collection logic.

    Args:
        base_env (BaseEnv): env implementing BaseEnv.
        extra_batch_callback (fn): function to send extra batch data to.
        policies (dict): Map of policy ids to PolicyGraph instances.
        policy_mapping_fn (func): Function that maps agent ids to policy ids.
            This is called when an agent first enters the environment. The
            agent is then "bound" to the returned policy for the episode.
        unroll_length (int): Number of episode steps before `SampleBatch` is
            yielded. Set to infinity to yield complete episodes.
        horizon (int): Horizon of the episode.
        preprocessors (dict): Map of policy id to preprocessor for the
            observations prior to filtering.
        obs_filters (dict): Map of policy id to filter used to process
            observations for the policy.
        clip_rewards (bool): Whether to clip rewards before postprocessing.
        pack (bool): Whether to pack multiple episodes into each batch. This
            guarantees batches will be exactly `unroll_length` in size.
        clip_actions (bool): Whether to clip actions to the space range.
        callbacks (dict): User callbacks to run on episode events.
        tf_sess (Session|None): Optional tensorflow session to use for batching
            TF policy evaluations.

    Yields:
        rollout (SampleBatch): Object containing state, action, reward,
            terminal condition, and other fields as dictated by `policy`.
    """

    try:
        if not horizon:
            horizon = (base_env.get_unwrapped()[0].spec.max_episode_steps)
    except Exception:
        logger.debug("no episode horizon specified, assuming inf")
    if not horizon:
        horizon = float("inf")

    # Pool of batch builders, which can be shared across episodes to pack
    # trajectory data.
    batch_builder_pool = []

    def get_batch_builder():
        if batch_builder_pool:
            return batch_builder_pool.pop()
        else:
            return MultiAgentSampleBatchBuilder(policies, clip_rewards)

    def new_episode():
        episode = MultiAgentEpisode(policies, policy_mapping_fn,
                                    get_batch_builder, extra_batch_callback)
        if callbacks.get("on_episode_start"):
            callbacks["on_episode_start"]({
                "env": base_env,
                "episode": episode
            })
        return episode

    active_episodes = defaultdict(new_episode)

    while True:
        # Get observations from all ready agents
        unfiltered_obs, rewards, dones, infos, off_policy_actions = \
            base_env.poll()

        # Process observations and prepare for policy evaluation
        active_envs, to_eval, outputs = _process_observations(
            base_env, policies, batch_builder_pool, active_episodes,
            unfiltered_obs, rewards, dones, infos, off_policy_actions, horizon,
            preprocessors, obs_filters, unroll_length, pack, callbacks)
        for o in outputs:
            yield o

        # Do batched policy eval
        eval_results = _do_policy_eval(tf_sess, to_eval, policies,
                                       active_episodes)

        # Process results and update episode state
        actions_to_send = _process_policy_eval_results(
            to_eval, eval_results, active_episodes, active_envs,
            off_policy_actions, policies, clip_actions)

        # Return computed actions to ready envs. We also send to envs that have
        # taken off-policy actions; those envs are free to ignore the action.
        base_env.send_actions(actions_to_send)


def _process_observations(base_env, policies, batch_builder_pool,
                          active_episodes, unfiltered_obs, rewards, dones,
                          infos, off_policy_actions, horizon, preprocessors,
                          obs_filters, unroll_length, pack, callbacks):
    """Record new data from the environment and prepare for policy evaluation.

    Returns:
        active_envs: set of non-terminated env ids
        to_eval: map of policy_id to list of agent PolicyEvalData
        outputs: list of metrics and samples to return from the sampler
    """

    active_envs = set()
    to_eval = defaultdict(list)
    outputs = []

    # For each environment
    for env_id, agent_obs in unfiltered_obs.items():
        new_episode = env_id not in active_episodes
        episode = active_episodes[env_id]
        if not new_episode:
            episode.length += 1
            episode.batch_builder.count += 1
            episode._add_agent_rewards(rewards[env_id])

        global _large_batch_warned
        if (not _large_batch_warned and
                episode.batch_builder.total() > max(1000, unroll_length * 10)):
            _large_batch_warned = True
            logger.warning(
                "More than {} observations for {} env steps ".format(
                    episode.batch_builder.total(),
                    episode.batch_builder.count) + "are buffered in "
                "the sampler. If this is more than you expected, check that "
                "that you set a horizon on your environment correctly. Note "
                "that in multi-agent environments, `sample_batch_size` sets "
                "the batch size based on environment steps, not the steps of "
                "individual agents, which can result in unexpectedly large "
                "batches.")

        # Check episode termination conditions
        if dones[env_id]["__all__"] or episode.length >= horizon:
            all_done = True
            atari_metrics = _fetch_atari_metrics(base_env)
            if atari_metrics is not None:
                for m in atari_metrics:
                    outputs.append(
                        m._replace(custom_metrics=episode.custom_metrics))
            else:
                outputs.append(
                    RolloutMetrics(episode.length, episode.total_reward,
                                   dict(episode.agent_rewards),
                                   episode.custom_metrics))
        else:
            all_done = False
            active_envs.add(env_id)

        # For each agent in the environment
        for agent_id, raw_obs in agent_obs.items():
            policy_id = episode.policy_for(agent_id)
            prep_obs = _get_or_raise(preprocessors,
                                     policy_id).transform(raw_obs)
            filtered_obs = _get_or_raise(obs_filters, policy_id)(prep_obs)
            agent_done = bool(all_done or dones[env_id].get(agent_id))
            if not agent_done:
                to_eval[policy_id].append(
                    PolicyEvalData(env_id, agent_id, filtered_obs,
                                   infos[env_id].get(agent_id, {}),
                                   episode.rnn_state_for(agent_id),
                                   episode.last_action_for(agent_id),
                                   rewards[env_id][agent_id] or 0.0))

            last_observation = episode.last_observation_for(agent_id)
            episode._set_last_observation(agent_id, filtered_obs)
            episode._set_last_raw_obs(agent_id, raw_obs)
            episode._set_last_info(agent_id, infos[env_id].get(agent_id, {}))

            # Record transition info if applicable
            if (last_observation is not None and infos[env_id].get(
                    agent_id, {}).get("training_enabled", True)):
                episode.batch_builder.add_values(
                    agent_id,
                    policy_id,
                    t=episode.length - 1,
                    eps_id=episode.episode_id,
                    agent_index=episode._agent_index(agent_id),
                    obs=last_observation,
                    actions=episode.last_action_for(agent_id),
                    rewards=rewards[env_id][agent_id],
                    prev_actions=episode.prev_action_for(agent_id),
                    prev_rewards=episode.prev_reward_for(agent_id),
                    dones=agent_done,
                    infos=infos[env_id].get(agent_id, {}),
                    new_obs=filtered_obs,
                    **episode.last_pi_info_for(agent_id))

        # Invoke the step callback after the step is logged to the episode
        if callbacks.get("on_episode_step"):
            callbacks["on_episode_step"]({"env": base_env, "episode": episode})

        # Cut the batch if we're not packing multiple episodes into one,
        # or if we've exceeded the requested batch size.
        if episode.batch_builder.has_pending_data():
            if (all_done and not pack) or \
                    episode.batch_builder.count >= unroll_length:
                outputs.append(episode.batch_builder.build_and_reset(episode))
            elif all_done:
                # Make sure postprocessor stays within one episode
                episode.batch_builder.postprocess_batch_so_far(episode)

        if all_done:
            # Handle episode termination
            batch_builder_pool.append(episode.batch_builder)
            if callbacks.get("on_episode_end"):
                callbacks["on_episode_end"]({
                    "env": base_env,
                    "episode": episode
                })
            del active_episodes[env_id]
            resetted_obs = base_env.try_reset(env_id)
            if resetted_obs is None:
                # Reset not supported, drop this env from the ready list
                if horizon != float("inf"):
                    raise ValueError(
                        "Setting episode horizon requires reset() support "
                        "from the environment.")
            else:
                # Creates a new episode
                episode = active_episodes[env_id]
                for agent_id, raw_obs in resetted_obs.items():
                    policy_id = episode.policy_for(agent_id)
                    policy = _get_or_raise(policies, policy_id)
                    prep_obs = _get_or_raise(preprocessors,
                                             policy_id).transform(raw_obs)
                    filtered_obs = _get_or_raise(obs_filters,
                                                 policy_id)(prep_obs)
                    episode._set_last_observation(agent_id, filtered_obs)
                    to_eval[policy_id].append(
                        PolicyEvalData(
                            env_id, agent_id, filtered_obs,
                            episode.last_info_for(agent_id) or {},
                            episode.rnn_state_for(agent_id),
                            np.zeros_like(
                                _flatten_action(policy.action_space.sample())),
                            0.0))

    return active_envs, to_eval, outputs


def _do_policy_eval(tf_sess, to_eval, policies, active_episodes):
    """Call compute actions on observation batches to get next actions.

    Returns:
        eval_results: dict of policy to compute_action() outputs.
    """

    eval_results = {}

    if tf_sess:
        builder = TFRunBuilder(tf_sess, "policy_eval")
        pending_fetches = {}
    else:
        builder = None
    for policy_id, eval_data in to_eval.items():
        rnn_in_cols = _to_column_format([t.rnn_state for t in eval_data])
        policy = _get_or_raise(policies, policy_id)
        if builder and (policy.compute_actions.__code__ is
                        TFPolicyGraph.compute_actions.__code__):
            # TODO(ekl): how can we make info batch available to TF code?
            pending_fetches[policy_id] = policy._build_compute_actions(
                builder, [t.obs for t in eval_data],
                rnn_in_cols,
                prev_action_batch=[t.prev_action for t in eval_data],
                prev_reward_batch=[t.prev_reward for t in eval_data])
        else:
            eval_results[policy_id] = policy.compute_actions(
                [t.obs for t in eval_data],
                rnn_in_cols,
                prev_action_batch=[t.prev_action for t in eval_data],
                prev_reward_batch=[t.prev_reward for t in eval_data],
                info_batch=[t.info for t in eval_data],
                episodes=[active_episodes[t.env_id] for t in eval_data])
    if builder:
        for k, v in pending_fetches.items():
            eval_results[k] = builder.get(v)

    return eval_results


def _process_policy_eval_results(to_eval, eval_results, active_episodes,
                                 active_envs, off_policy_actions, policies,
                                 clip_actions):
    """Process the output of policy neural network evaluation.

    Records policy evaluation results into the given episode objects and
    returns replies to send back to agents in the env.

    Returns:
        actions_to_send: nested dict of env id -> agent id -> agent replies.
    """

    actions_to_send = defaultdict(dict)
    for env_id in active_envs:
        actions_to_send[env_id] = {}  # at minimum send empty dict

    for policy_id, eval_data in to_eval.items():
        rnn_in_cols = _to_column_format([t.rnn_state for t in eval_data])
        actions, rnn_out_cols, pi_info_cols = eval_results[policy_id]
        if len(rnn_in_cols) != len(rnn_out_cols):
            raise ValueError("Length of RNN in did not match RNN out, got: "
                             "{} vs {}".format(rnn_in_cols, rnn_out_cols))
        # Add RNN state info
        for f_i, column in enumerate(rnn_in_cols):
            pi_info_cols["state_in_{}".format(f_i)] = column
        for f_i, column in enumerate(rnn_out_cols):
            pi_info_cols["state_out_{}".format(f_i)] = column
        # Save output rows
        actions = _unbatch_tuple_actions(actions)
        policy = _get_or_raise(policies, policy_id)
        for i, action in enumerate(actions):
            env_id = eval_data[i].env_id
            agent_id = eval_data[i].agent_id
            if clip_actions:
                actions_to_send[env_id][agent_id] = _clip_actions(
                    action, policy.action_space)
            else:
                actions_to_send[env_id][agent_id] = action
            episode = active_episodes[env_id]
            episode._set_rnn_state(agent_id, [c[i] for c in rnn_out_cols])
            episode._set_last_pi_info(
                agent_id, {k: v[i]
                           for k, v in pi_info_cols.items()})
            if env_id in off_policy_actions and \
                    agent_id in off_policy_actions[env_id]:
                episode._set_last_action(agent_id,
                                         off_policy_actions[env_id][agent_id])
            else:
                episode._set_last_action(agent_id, action)

    return actions_to_send


def _fetch_atari_metrics(base_env):
    """Atari games have multiple logical episodes, one per life.

    However for metrics reporting we count full episodes all lives included.
    """
    unwrapped = base_env.get_unwrapped()
    if not unwrapped:
        return None
    atari_out = []
    for u in unwrapped:
        monitor = get_wrapper_by_cls(u, MonitorEnv)
        if not monitor:
            return None
        for eps_rew, eps_len in monitor.next_episode_results():
            atari_out.append(RolloutMetrics(eps_len, eps_rew, {}, {}))
    return atari_out


def _clip_actions(actions, space):
    """Called to clip actions to the specified range of this policy.

    Arguments:
        actions: Single action.
        space: Action space the actions should be present in.

    Returns:
        Clipped batch of actions.
    """

    if isinstance(space, gym.spaces.Box):
        return np.clip(actions, space.low, space.high)
    elif isinstance(space, gym.spaces.Tuple):
        if type(actions) not in (tuple, list):
            raise ValueError("Expected tuple space for actions {}: {}".format(
                actions, space))
        out = []
        for a, s in zip(actions, space.spaces):
            out.append(_clip_actions(a, s))
        return out
    else:
        return actions


def _unbatch_tuple_actions(action_batch):
    # convert list of batches -> batch of lists
    if isinstance(action_batch, TupleActions):
        out = []
        for j in range(len(action_batch.batches[0])):
            out.append([
                action_batch.batches[i][j]
                for i in range(len(action_batch.batches))
            ])
        return out
    return action_batch


def _to_column_format(rnn_state_rows):
    num_cols = len(rnn_state_rows[0])
    return [[row[i] for row in rnn_state_rows] for i in range(num_cols)]


def _get_or_raise(mapping, policy_id):
    if policy_id not in mapping:
        raise ValueError(
            "Could not find policy for agent: agent policy id `{}` not "
            "in policy map keys {}.".format(policy_id, mapping.keys()))
    return mapping[policy_id]
