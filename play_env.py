from wrappers.common_wrappers import VisualObservationWrapper, \
    JumpAfterPlace, Discretization, ColorWrapper, StorePrevObsWrapper, InitVarsWrapper, WithoutColors, SingleActiveAction
from wrappers.loggers import SuccessRateFullFigure, VideoLogger, SuccessRateLogger, CompletedRateLogger
#from wrappers.multitask import TargetGenerator, SubtaskGenerator
from wrappers.reward_wrappers import RangetRewardFilledField, Closeness, PutUnderReward, OneBlockReward
from wrappers.target_generator import RandomFigure

from wrappers.EpisodeController import *

import readchar

class WI():
    def __init__(self):
        pass

    def init_world(self, target):
        start_grid = np.zeros((9, 11, 11))
        pos = (0, 3, 0, 0, 0)
        return start_grid, pos, target

def make_flying_iglu(*args, **kwargs):
    from gridworld.env import GridWorld
    from gridworld.tasks.task import Task
    env = GridWorld(render=True, render_size=(512, 512), select_and_place=True, action_space='flying', discretize=True, max_steps=1000, fake=kwargs.get('fake', False))
    env.set_task(Task("", np.ones((9, 11, 11)), invariant=False))

    tg = RandomTargetGenerator(None, 0.01)
    sg = FlyingSubtaskGenerator()
    #sg = NavigationSubtaskGenerator()
    target = tg.get_target()
    tc = TrainTaskController()
    sc = TrainSubtaskController()
    #wi = NavigationWorldInitializer()
    wi = WI()

    env = InitVarsWrapper(env, tg, sg, tc, sc, wi)

    env = Discretization(env)
    env = WithoutColors(env)
    env = SingleActiveAction(env)

    env = StorePrevObsWrapper(env)

    env = OneBlockReward(env)
    env = Closeness(env)

    #env = VisualObservationWrapper(env)
    env = EpisodeController(env, tg, sg, tc, sc, wi)
    env = SuccessRateLogger(env)

    #env = CompletedRateLogger(env)
    return env

env = make_flying_iglu()
env.reset()

def map_act(c):
    if c >= '0' and c <= '9':
        return int(c)
    s = 'qwerty'
    return s.index(c) + 10
import numpy as np
while True:
    obs, reward, done, info = env.step(map_act(readchar.readkey()))
    print(reward)
    print(np.nonzero(env.task.target_grid), env.task.target_grid[np.nonzero(env.task.target_grid)])
    print('-------')
    if done:
        print('-----------------done-----------------')
        print(reward)
        print(info['episode_extra_stats']['SuccessRate'])
        #print(info['episode_extra_stats']['CompletedRate'])
        env.reset()