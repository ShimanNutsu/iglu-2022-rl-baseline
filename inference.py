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

class WI():
    def __init__(self):
        pass

    def init_world(self, target):
        start_grid = np.zeros((9, 11, 11))
        x = np.random.choice(np.arange(-5, 5))
        y = np.random.choice(np.arange(-5, 5))
        z = np.random.choice(np.arange(6))
        pos = (x, z, y, 0, 0)
        return start_grid, pos, target

from dataset_loader import get_grids

class DatasetTargetGenerator(TargetGenerator):
    def __init__(self, a, b):
        self.targets = get_grids()[1]
        self.cur = 0
        pass

    def get_target(self):
        self.cur += 1
        return self.targets[self.cur - 1]

def make_flying_iglu(*args, **kwargs):
    from gridworld.env import GridWorld
    from gridworld.tasks.task import Task
    env = GridWorld(render=True, select_and_place=True, action_space='flying', discretize=True, max_steps=1000, fake=kwargs.get('fake', False))
    env.set_task(Task("", np.ones((9, 11, 11)), invariant=False))

    tg = RandomTargetGenerator(None, 0.01)
    sg = FlyingSubtaskGenerator()
    #sg = NavigationSubtaskGenerator()
    target = tg.get_target()
    tc = TestTaskController()
    sc = TrainSubtaskController()
    wi = WI()

    env = InitVarsWrapper(env, tg, sg, tc, sc, wi)

    env = Discretization(env)
    env = WithoutColors(env)
    env = SingleActiveAction(env)

    env = StorePrevObsWrapper(env)

    env = OneBlockReward(env)
    env = Closeness(env)

    env = VisualObservationWrapper(env)
    env = EpisodeController(env, tg, sg, tc, sc, wi)
    #env = VideoLogger(env)
    #env = SuccessRateLogger(env)

    #env = CompletedRateLogger(env)

    #env = MultiAgentWrapper(env)
    #env = AutoResetWrapper(env)
    return env

def make_iglu(*args, **kwargs):
    return make_flying_iglu(*args, **kwargs)

from pretr_agent import make_agent
from metrics import get_metrics

if __name__ == '__main__':
    hor_angle = 0
    vert_angle = 0

    agent = make_agent()
    env = make_iglu()

    targets, preds = get_grids()
    builts = []

    success = 0
    num_eps = 84
    metrics = []
    m1= []
    m2 = []
    for j in range(num_eps):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act([obs])[0]

            if action in [6, 7]:
                hor_angle += (1 if action == 6 else -1)
            if action in [8, 9]:
                vert_angle += (1 if action == 8 else -1)

            obs, rew, done, info = env.step(action)
            if action == 11 and not done:
                for i in range(18):
                    action = 7# if hor_angle > 0 else 6
                    obs, rew, done, info = env.step(action)
                
                #hor_angle = 0

                #for i in range(abs(vert_angle)):
                #    action = 9 if vert_angle > 0 else 8
                #    obs, rew, done, info = env.step(action)
                #for i in range(20):
                #    obs, rew, done, info = env.step(0)
            if done:
                builts.append(obs['grid'])
                print('***********************', env.subtask_generator.empty())
                if env.subtask_generator.empty():
                    success += 1
                #metrics.append(get_metrics(env.task.target_grid, obs['grid'])[1])
                m1.append(get_metrics(targets[j], obs['grid'])[1])
                m2.append(get_metrics(env.task.target_grid, obs['grid'])[1])
        print('\n\n ' + str(j) + ' \n\n')
    #np.save('targets', np.array(targets))
    #np.save('preds', np.array(preds))
    #np.save('builts', np.array(builts))
    print(success / num_eps)
    print(metrics)
    print(np.mean(m1))
    print(np.mean(m2))