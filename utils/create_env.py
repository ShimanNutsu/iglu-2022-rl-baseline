import gym
import numpy as np

from gym.spaces import Box
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper

import sys
sys.path.append("./")
from wrappers.common_wrappers import VisualObservationWrapper, \
    JumpAfterPlace, Discretization, ColorWrapper, StorePrevObsWrapper, InitVarsWrapper, WithoutColors, SingleActiveAction
from wrappers.loggers import SuccessRateFullFigure, VideoLogger, SuccessRateLogger, CompletedRateLogger
#from wrappers.multitask import TargetGenerator, SubtaskGenerator
from wrappers.reward_wrappers import RangetRewardFilledField, Closeness, PutUnderReward, OneBlockReward
from wrappers.target_generator import RandomFigure

from wrappers.EpisodeController import *



class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        if all(dones):
            observations = self.env.reset()
        return observations, rewards, dones, infos


class FakeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space
        self.observation_space['obs'] = Box(0.0, 1.0, shape=(1,))

    def observation(self, observation):
        observation['obs'] = np.array([0.0])
        return observation

def make_walking_iglu(*args, **kwargs):
    from gridworld.env import GridWorld
    from gridworld.tasks.task import Task
    env = GridWorld(render=True, select_and_place=True, discretize=True, max_steps=1000, fake=kwargs.get('fake', False))
    env.set_task(Task("", np.ones((9, 11, 11)), invariant=False))

    tg = RandomTargetGenerator(None, 0.01)
    sg = FlyingSubtaskGenerator()
    #sg = NavigationSubtaskGenerator()
    target = tg.get_target()
    sg.set_new_task(target)
    tc = TrainTaskController()
    sc = TrainSubtaskController()
    wi = WorldInitializer()

    env = InitWrapper(env, tg, sg, tc, sc, wi)

    env = StorePrevObsWrapper(env)

    env = JumpAfterPlace(env)
    env = OneBlockReward(env)
    env = Closeness(env)
    env = PutUnderReward(env)
    env = SuccessRateLogger(env)
    env = CompletedRateLogger(env)

    env = VisualObservationWrapper(env)

    env = WithoutColors(env)
    env = SingleActiveAction(env)

    env = EpisodeController(env, tg, sg, tc, sc, wi)

    env = MultiAgentWrapper(env)
    env = AutoResetWrapper(env)
    return env

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
    env = SuccessRateLogger(env)

    env = VisualObservationWrapper(env)
    env = EpisodeController(env, tg, sg, tc, sc, wi)

    env = CompletedRateLogger(env)

    env = MultiAgentWrapper(env)
    env = AutoResetWrapper(env)
    return env

def make_iglu(*args, **kwargs):
    return make_flying_iglu(*args, **kwargs)

if __name__ == '__main__':
    env = make_iglu()
    env.reset()
    env.step(10)