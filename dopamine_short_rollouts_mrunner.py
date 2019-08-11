"""Implementation of Runner and Agent for Dopamine's Rainbow with short rollouts"""
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains.run_experiment import Runner
from dopamine.agents.rainbow.rainbow_agent import RainbowAgent
from dopamine.agents.dqn import dqn_agent
from mrunner.helpers.client_helper import logger, get_configuration
from math import exp

import tensorflow as tf
import numpy as np
import gin


@gin.configurable
def default_rollout_sampler(steps, since_last_rollout, exponential_coefficient=0.05):
    """ Default choice for rollout sampler used in RolloutsRunner
    :param steps: number of steps in the current episode
    :param since_last_rollout: number of steps since last rollout
    :param exponential_coefficent: controls rate at which probability of chosing given state
    increases.
    :return:
    """
    return 1 - exp(-since_last_rollout*exponential_coefficient)


@gin.configurable
class RolloutsRunner(Runner):
    """Implementation of short rollouts, based on Runner. Has to be used with RainbowRolloutsAgent as agent"""
    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 create_environment_fn=atari_lib.create_atari_environment,
                 checkpoint_file_prefix='ckpt',
                 logging_file_prefix='log',
                 log_every_n=1,
                 num_iterations=200,
                 training_steps=250000,
                 evaluation_steps=125000,
                 max_steps_per_episode=27000,
                 rollout_sampler=default_rollout_sampler,
                 rollout_len=15,
                 logg = logger):
        """
        :param rollout_sampler: function which outputs probability that a given state will be chosen for rollout base
        states. Takes number of steps in the current episode and steps since last rollout as arguments
        :param rollout_len: constant used as short rollouts maximal length
        :param logg: function for logging: takes (metric,value) as arguments, defaults to mrunner's logger
        """
        super(RolloutsRunner, self).__init__(base_dir, create_agent_fn, create_environment_fn, checkpoint_file_prefix,
                                             logging_file_prefix, log_every_n, num_iterations, training_steps,
                                             evaluation_steps, max_steps_per_episode)
        self._rollout_sampler = rollout_sampler
        self._rollout_len = rollout_len
        self.logger = logg

        self._global_steps = 0
        self._global_rollout_steps = 0

    def _run_short_rollout(self, base_state):
        """Runs short rollout using environment, starting from base state
        :param base_state:
        :return:
        """
        # We change agent's mode to short rollout so that transitions will be stored
        # and agent's train op will be run
        self._agent.main_trajectory = False

        step_number = 0
        total_reward = 0
        # We have to switch epsilon settings to rollout
        self._agent.switch_epsilon_settings(main=False)

        env_cpy, obs, length = base_state
        self._environment.environment.unwrapped.restore_full_state(env_cpy)
        action = self._agent.begin_episode(obs)

        while True:
            observation, reward, is_terminal = self._run_one_step(action)
            total_reward += reward
            step_number += 1
            reward = np.clip(reward, -1, 1)

            if(is_terminal or step_number >= length):
                # We cut rollout without setting terminal flag
                # and storing final transition
                break

            action = self._agent.step(reward, observation)

        # We switch agent's settings back to main trajectory mode:
        self._agent.main_trajectory = True

        return step_number, total_reward

    def _run_one_episode(self):
        """Runs one episode of environment and short rollouts collected during that episode
        :return:
        """
        assert self._agent.main_trajectory

        step_number = 0
        total_reward = 0
        base_states = []
        since_last_rollout = 0
        rollout_steps = 0
        rollout_rewards = 0

        # We have to switch epsilon settings to main
        self._agent.switch_epsilon_settings()

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal = self._run_one_step(action)
            total_reward += reward
            step_number += 1
            # Perform reward clipping.
            reward = np.clip(reward, -1, 1)

            if (self._environment.game_over or
                    step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode.
                break
            elif is_terminal:
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._agent.end_episode(reward)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation)
                if not self._agent.eval_mode:
                    # If we are in training mode, we add current state to base_states
                    # for future rollouts (with probability specified by self._rollout_sampler)
                    prob = self._rollout_sampler(step_number, since_last_rollout)
                    if prob >= np.random.random():
                        rollout_len = self._rollout_len
                        base_states.append((self._environment.environment.unwrapped.clone_full_state(),
                                            observation, rollout_len))
                        since_last_rollout = 0

                    else:
                        since_last_rollout += 1

        self._end_episode(reward)
        # For every base state we create short rollout
        for base_state in base_states:
            steps, reward = self._run_short_rollout(base_state)
            rollout_steps += steps
            rollout_rewards += reward

        if not self._agent.eval_mode:
            self.logger('episode reward', total_reward)
            self.logger('rollout steps per episode', rollout_steps)
            self.logger('rollouts per episode', len(base_states))
            self._global_steps += step_number
            self._global_rollout_steps += rollout_steps
            self.logger('main steps up to episode', self._global_steps)
            self.logger('rollout steps up to episode', self._global_rollout_steps)

        else:
            self.logger('eval episode reward', total_reward)
            self.logger('main steps up to eval episode', self._global_steps)
            self.logger('rollout steps up to eval episode', self._global_rollout_steps)

        return step_number, total_reward


@gin.configurable
class RainbowRolloutsAgent(RainbowAgent):
    """Implementation of RainbowAgent which allows for the usage of different epsilons
    for main branch and short rollouts"""
    def __init__(self,
                 sess,
                 num_actions,
                 observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
                 stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
                 network=atari_lib.rainbow_network,
                 num_atoms=51,
                 vmax=10.,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 epsilon_rollout_fn=dqn_agent.linearly_decaying_epsilon,
                 epsilon_rollout_train=0.1,
                 epsilon_rollout_decay_period=10**6,
                 replay_scheme='prioritized',
                 tf_device='/cpu:*',
                 use_staging=True,
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.00025, epsilon=0.0003125),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 ):
        super().__init__(sess,num_actions,observation_shape,observation_dtype,stack_size,network,num_atoms,vmax,
                         gamma,update_horizon,min_replay_history,update_period,target_update_period,epsilon_fn,
                         epsilon_train,epsilon_eval,epsilon_decay_period,replay_scheme,tf_device,use_staging,
                         optimizer,summary_writer,summary_writing_frequency)

        # We make epsilon_train and epsilon_fn state-dependent:
        # They will take on different values in main rollout and branch when switched by self.switch_epsilons_settings

        self._epsilon_main_fn = epsilon_fn
        self._epsilon_rollout_fn = epsilon_rollout_fn
        self._epsilon_main_train = epsilon_train
        self._epsilon_rollout_train = epsilon_rollout_train
        self._epsilon_main_decay_period = epsilon_decay_period
        self._epsilon_rollout_decay_period = epsilon_rollout_decay_period

        # This will indicate whether we are in main trajectory
        self.main_trajectory = True

    def switch_epsilon_settings(self,main=True):
        """Switches epsilon settings to main if main=True and to rollout otherwise
        :return:
        """
        if main:
            self.epsilon_fn = self._epsilon_main_fn
            self.epsilon_train = self._epsilon_main_train
            self.epsilon_decay_period = self._epsilon_main_decay_period
        else:
            self.epsilon_fn = self._epsilon_rollout_fn
            self.epsilon_train = self._epsilon_rollout_train
            self.epsilon_decay_period = self._epsilon_rollout_decay_period

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.
        We store the observation of the last time step since we want to store it
        with the reward.
        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.
        Returns:
          int, the selected action.
        """
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode and not self.main_trajectory:
            self._store_transition(self._last_observation, self.action, reward, False)
            self._train_step()

        self.action = self._select_action()
        return self.action


def create_rainbow_rollouts_agent(sess, environment, summary_writer=None):
    return RainbowRolloutsAgent(sess, num_actions=environment.action_space.n,
                                summary_writer=summary_writer)


def main():
    """Follows example from mrunner experiment_gin.py
    Requires LOG_PATH to be passed in config
    :return:
    """
    params = get_configuration(print_diagnostics=True, with_neptune=True, inject_parameters_to_gin=True)
    runner = RolloutsRunner(params.LOG_PATH,create_rainbow_rollouts_agent)
    runner.run_experiment()


if __name__ == '__main__':
    main()
