import sys
import json
import os
from os.path import join

import torch

from utils.config_validation import Experiment
from argparse import Namespace
from pathlib import Path
import numpy as np
import wandb
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES

from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint

from models.models import ResnetEncoderWithTarget
from gridworld.env import GridWorld
from gridworld.tasks.task import Task
import gym
from gym.spaces import Box



from wrappers.EpisodeController import *
from wrappers.common_wrappers import *
from wrappers.reward_wrappers import *
from wrappers.loggers import *

# from wrappers import RandomFigure, TargetGenerator, SubtaskGenerator, VectorObservationWrapper, JumpAfterPlace, FakeObsWrapper


class EndActionController(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.action_space.n + 1)


def make_flying_iglu(*args, **kwargs):
    from gridworld.env import GridWorld
    from gridworld.tasks.task import Task
    env = GridWorld(render=True, select_and_place=True, action_space='flying', discretize=True, max_steps=1000, fake=kwargs.get('fake', False))
    env.set_task(Task("", np.ones((9, 11, 11)), invariant=False))

    tg = RandomTargetGenerator(None, 0.01)
    #sg = FlyingSubtaskGenerator()
    sg = NavigationSubtaskGenerator()
    target = tg.get_target()
    tc = TrainTaskController()
    sc = TrainSubtaskController()
    wi = NavigationWorldInitializer()

    env = InitVarsWrapper(env, tg, sg, tc, sc, wi)

    env = Discretization(env)
    env = WithoutColors(env)
    env = SingleActiveAction(env)

    env = StorePrevObsWrapper(env)

    env = OneBlockReward(env)
    env = Closeness(env)

    env = VisualObservationWrapper(env)
    env = EpisodeController(env, tg, sg, tc, sc, wi)
    #env = SuccessRateLogger(env)

    #env = CompletedRateLogger(env)

    #env = MultiAgentWrapper(env)
    #env = AutoResetWrapper(env)
    return env

def make_iglu(*args, **kwargs):
    return make_flying_iglu(*args, **kwargs)


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='IGLUSilentBuilder-v0',
        make_env_func=make_iglu,
    )
    register_custom_encoder('custom_env_encoder', ResnetEncoderWithTarget)


# EXTRA_PER_POLICY_SUMMARIES.append(iglu_extra_summaries)

def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            full_config=exp.dict()
                            )
    return exp, flat_config


class APPOHolder:
    def __init__(self, algo_cfg):
        self.cfg = algo_cfg

        path = algo_cfg.path_to_weights
        device = algo_cfg.device
        register_custom_components()

        self.path = path
        self.env = None
        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        exp, flat_config = validate_config(config['full_config'])
        algo_cfg = flat_config

        env = create_env(algo_cfg.env, cfg=algo_cfg, env_config={})

        actor_critic = create_actor_critic(algo_cfg, env.observation_space, env.action_space)
        env.close()

        if device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        self.device = device

        # actor_critic.share_memory()
        actor_critic.model_to_device(device)
        policy_id = algo_cfg.policy_index
        checkpoints = join(path, f'checkpoint_p{policy_id}')
        checkpoints = LearnerWorker.get_checkpoints(checkpoints)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = algo_cfg

        self.rnn_states = None

    def after_reset(self, env):
        self.env = env

    @staticmethod
    def get_additional_info():
        return {"rl_used": 1.0}

    def act(self, observations, rewards=None, dones=None, infos=None):

        if self.rnn_states is None or len(self.rnn_states) != len(observations):
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)

        with torch.no_grad():
            #  print(self.rnn_states)
            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)
            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        actions = actions.cpu().numpy()
        if actions[0] == 11:
            self.clear_hidden()
        return actions

    def clear_hidden(self):
        self.rnn_states = None


def make_agent():
    register_custom_components()

    cfg = parse_args(argv=[
        '--algo=APPO', '--env=IGLUSilentBuilder-v0', '--experiment=TreeChopBaseline-iglu',
        '--experiments_root=./',
        '--train_dir=./train_dir/0013'
    ],
        evaluation=True)
    cfg = load_from_checkpoint(cfg)

    cfg.setdefault("path_to_weights", "./train_dir/0013/TreeChopBaseline-iglu")
    return APPOHolder(cfg)
