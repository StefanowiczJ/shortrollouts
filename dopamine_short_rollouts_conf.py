from mrunner.helpers.specification_helper import create_experiments_helper
import os

if "NEPTUNE_API_TOKEN" not in os.environ or "PROJECT_QUALIFIED_NAME" not in os.environ:
    print("Please set NEPTUNE_API_TOKEN and PROJECT_QUALIFIED_NAME env variables")
    print("Their values can be from up.neptune.ml. Click help and then quickstart.")
    exit()

base_config = {
    'atari_lib.create_atari_environment.sticky_actions': True,
    'BASE_PATH': '.'
}
params_grid = {
    'GAME': ['Asterix', 'Breakout', 'Pong'],
    'dopamine_short_rollouts.default_rollout_sampler.exponential_coefficient': [0.1, 0.05],
    'dopamine_short_rollouts.RolloutsRunner.rollout_len': [20, 10],
    'dopamine_short_rollouts.RaibowRolloutsAgent.epsilon_train': [0.05, 0.01],
    'dopamine_short_rollouts.RaibowRolloutsAgent.epsilon_decay_period': [10**6, 250000],
    'dopamine_short_rollouts.RaibowRolloutsAgent.epsilon_rollout_train': [0.1, 0.01],
    'dopamine_short_rollouts.RaibowRolloutsAgent.epsilon_rollout_decay_period': [10**6, 250000]
}

experiments_list = create_experiments_helper(experiment_name='Gin experiment',
                                            project_name=os.environ["PROJECT_QUALIFIED_NAME"],
                                            script='python dopamine_short_rollouts_mrunner.py',
                                            python_path='.', paths_to_dump='',
                                            tags=[],
                                            base_config=base_config, params_grid=params_grid)
