from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains.run_experiment import Runner, create_agent
from mrunner.helpers.client_helper import logger, get_configuration

import numpy as np
import os


class RainbowBasicRunner(Runner):
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
                 logg = logger):
        """
        :param rollout_sampler: function which outputs probability that a given state will be chosen for rollout base
        states. Takes number of steps in the current episode and steps since last rollout as arguments
        :param rollout_len: constant used as short rollouts maximal length
        """
        super(RainbowBasicRunner, self).__init__(base_dir, create_agent_fn, create_environment_fn, checkpoint_file_prefix,
                                             logging_file_prefix, log_every_n, num_iterations, training_steps,
                                             evaluation_steps, max_steps_per_episode)
        self.logger = logg
        self._global_steps = 0

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.
        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.

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

        self._end_episode(reward)

        if not self._agent.eval_mode:
            self.logger('episode reward', total_reward)
            self._global_steps += step_number
            self.logger('main steps up to episode', self._global_steps)

        else:
            self.logger('eval episode reward', total_reward)
            self.logger('main steps up to eval episode', self._global_steps)

        return step_number, total_reward


def main():
    params = get_configuration(print_diagnostics=True, with_neptune=True, inject_parameters_to_gin=True)
    LOG_PATH = os.path.join(params.BASE_PATH, 'tests', params.GAME)
    runner = RainbowBasicRunner(LOG_PATH, create_agent)
    runner.run_experiment()

if __name__ == '__main__':
    main()
