import gym
import numpy as np

from gym.spaces import Box
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper

from wrappers.common_wrappers import VisualObservationWrapper, \
    JumpAfterPlace, Discretization, ColorWrapper
from wrappers.loggers import SuccessRateFullFigure, VideoLogger
#from wrappers.multitask import TargetGenerator, SubtaskGenerator
from wrappers.reward_wrappers import RangetRewardFilledField, Closeness
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


def make_iglu(*args, **kwargs):
    from gridworld.env import GridWorld
    from gridworld.tasks.task import Task
    custom_grid = np.ones((9, 11, 11))
    env = GridWorld(render=True, select_and_place=True, discretize=True, max_steps=900, fake=kwargs.get('fake', False))
    env.set_task(Task("", custom_grid, invariant=False))
    
    tg = RandomTargetGenerator(None, 0.01)
    sg = WalkingSubtaskGenerator()
    target = tg.get_target(None)
    sg.set_new_task(target)
    tc = TrainTaskController()
    sc = TrainSubtaskController()
    env = EpisodeController(env, tg, sg, tc, sc)
    #figure_generator = RandomFigure
    
    #env = TargetGenerator(env, fig_generator=figure_generator)
    #env = SubtaskGenerator(env)
    env = VisualObservationWrapper(env)
    env = JumpAfterPlace(env)

    #env = Discretization(env)
    #env = ColorWrapper(env)
    env = RangetRewardFilledField(env)
    env = Closeness(env)

    env = SuccessRateFullFigure(env)
    #env = VideoLogger(env)
    env = MultiAgentWrapper(env)
    env = AutoResetWrapper(env)

    return env
