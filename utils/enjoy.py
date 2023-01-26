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
from wrappers.common_wrappers import *
from wrappers.loggers import *
from wrappers.reward_wrappers import *
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
    
class CustomTarget(TargetGenerator):
    def get_target(self):
        grid = np.zeros((9, 11, 11))
        for i in range(4):
            grid[i, 3, 4] = 1
        grid[1, 3, 4] = 0
        for i in range(2):
            grid[i, 5, 5] = 1
        grid[0, 4, 7] = 1
        grid[2, 7, 6] = 1
        grid[0, 6, 6] = 1
        return grid

class WI():
    def __init__(self):
        pass

    def init_world(self, target):
        start_grid = np.zeros((9, 11, 11))
        pos = (0, 3, 0, 0, 0)
        return start_grid, pos, target

def makhe_iglu(*args, **kwargs):
    from gridworld.env import GridWorld
    from gridworld.tasks.task import Task
    env = GridWorld(render=True, select_and_place=True, action_space='flying', discretize=True, max_steps=1000, fake=kwargs.get('fake', False))
    env.set_task(Task("", np.ones((9, 11, 11)), invariant=False))

    tg = RandomTargetGenerator(None, 0.01)
    #tg = CustomTarget(0.01)
    #sg = FlyingSubtaskGenerator()
    sg = NavigationSubtaskGenerator()
    target = tg.get_target()
    tc = TrainTaskController()
    sc = TrainSubtaskController()
    wi = NavigationWorldInitializer()
    #wi = WI()

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

    env = VideoLogger(env)
    #env = MultiAgentWrapper(env)
    #env = AutoResetWrapper(env)
    return env


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='IGLUSilentBuilder-v0',
        make_env_func=makhe_iglu,
    )
    register_custom_encoder('custom_env_encoder', ResnetEncoderWithTarget)

#from create_env import make_iglu

def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args(argv=['--algo=APPO', '--env=IGLUSilentBuilder-v0', '--experiment=TreeChopBaseline-iglu',
                           '--experiments_root=./',
                           #'--experiments_root=force_envs_single_thread=False;num_envs_per_worker=1;num_workers=10',
                           '--train_dir=./train_dir/nav_from_high'], evaluation=True)
    status = enjoy(cfg, 5000)
    return status


if __name__ == '__main__':
    sys.exit(main())
