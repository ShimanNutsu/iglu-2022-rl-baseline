import sys

import numpy as np
from gridworld.env import GridWorld
from gridworld.tasks.task import Task
from sample_factory.algorithms.appo.enjoy_appo import enjoy
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.env_registry import global_env_registry

sys.path.append("./")
from models.models import ResnetEncoderWithTarget
from wrappers.common_wrappers import VisualObservationWrapper, \
    ColorWrapper, JumpAfterPlace, Discretization
from wrappers.loggers import VideoLogger, Logger, \
                    SuccessRateFullFigure#, StatisticsLogger, Statistics, R1_score
#from wrappers.multitask import SubtaskGenerator, TargetGenerator
from wrappers.reward_wrappers import RangetRewardFilledField, Closeness
from wrappers.EpisodeController import *

from utils.create_env import make_iglu

from wrappers.target_generator import  RandomFigure, CustomFigure

def tasks_from_database():
    names = np.load('data/augmented_target_name.npy')
    targets = np.load('data/augmented_targets.npy')    
    return dict(zip(names, targets))

def castom_tasks():
    tasks = dict()   
    
    t1 = np.zeros((9,11,11))
    t1[0, 1:4, 1:4] = 1
    tasks['[0, 1:4, 1:4]'] = t1
    
    t2 = np.zeros((9,11,11))
    t2[0:2, 1:4, 1:4] = 1
    tasks['[0:2, 1:4, 1:4]'] = t2
    
    t3 = np.zeros((9,11,11))
    t3[0:5, 4, 4] = 1
    t3[1, 4, 4] = 0
    t3[3, 4, 4] = 0
    t3[0, 8, 7] = 1
    tasks['[0:7, 4, 4]'] = t3
    
    t4 = np.zeros((9,11,11))
    t4[0, 4:8, 4:8] = 1
    tasks['[0, 4:8, 4:8]'] = t4
    
    t5 = np.zeros((9,11,11))
    t5[0:3, 8:10, 8:10] = 1
    tasks['[0:3, 8:10, 8:10]'] = t5
    return tasks
    
def makhe_iglu(*args, **kwargs):
    from gridworld.env import GridWorld
    from gridworld.tasks.task import Task
    custom_grid = np.ones((9, 11, 11))
    #env = GridWorld(render=True, select_and_place=True, discretize=True, action_space='flying', max_steps=1000,   fake=kwargs.get('fake', False))
    #env = GridWorld(render=True, select_and_place=True, discretize=True, max_steps=1000,   fake=kwargs.get('fake', False))
    env = GridWorld(render=True, select_and_place=True, discretize=True, max_steps=150,   fake=kwargs.get('fake', False))
    env.set_task(Task("", custom_grid, invariant=False))
    
    tg = RandomTargetGenerator(None, 0.01)
    #sg = WalkingSubtaskGenerator()
    sg = NavigationSubtaskGenerator()
    target = tg.get_target(None)
    sg.set_new_task(target)
    tc = TrainTaskController()
    sc = TrainSubtaskController()
    env = EpisodeController(env, tg, sg, tc, sc)
    #figure_generator = RandomFigure
    
    #env = TargetGenerator(env, fig_generator=figure_generator)
    #env = SubtaskGenerator(env)
    env = JumpAfterPlace(env)
    env = VisualObservationWrapper(env)

    #env = Discretization(env)
    #env = ColorWrapper(env)
    env = RangetRewardFilledField(env)
    env = Closeness(env)

    env = SuccessRateFullFigure(env)
    env = VideoLogger(env)
    #env = MultiAgentWrapper(env)
    #env = AutoResetWrapper(env)

    return env


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='IGLUSilentBuilder-v0',
        make_env_func=make_iglu,
    )
    register_custom_encoder('custom_env_encoder', ResnetEncoderWithTarget)

#from create_env import make_iglu

def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args(argv=['--algo=APPO', '--env=IGLUSilentBuilder-v0', '--experiment=TreeChopBaseline-iglu',
                           '--experiments_root=./',
                           #'--experiments_root=force_envs_single_thread=False;num_envs_per_worker=1;num_workers=10',
                           '--train_dir=./train_dir/0022'], evaluation=True)
    status = enjoy(cfg, 5000)
    return status


if __name__ == '__main__':
    sys.exit(main())
