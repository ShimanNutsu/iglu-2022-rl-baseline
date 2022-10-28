import gym
from gridworld.tasks import Task
import numpy as np

class TargetGenerator:
    def __init__(self, config):
        pass
    
    def get_target(self, obs):
        raise NotImplementedError("Using virtual class " + str(self.__class__.__name__))

class RandomTargetGenerator(TargetGenerator):
    def __init__(self, config, p):
        self.p = p
        pass

    def get_target(self, obs):
        return np.random.multinomial(6, [1 - self.p] + [self.p / 6] * 6, (9, 11, 11))

#---------------------------------------------

class TaskController():
    def __init__(self) -> None:
        pass

    def finished(self, subtask_controller, obs, prev_obs):
        raise NotImplementedError("Using virtual class " + str(self.__class__.__name__))

class TrainTaskController(TaskController):
    def __init__(self) -> None:
        pass

    def finished(self, subtask_generator, obs, prev_obs):
        placed = obs['inventary'] != prev_obs['inventory']
        if subtask_generator.empty():
            return True
        if placed and obs['grid'] - prev_obs['grid'] != subtask_generator.current_task:
            return True
        return False

#---------------------------------------------

class SubtaskController():
    def __init__(self) -> None:
        pass

    def finished(self, subtask_generator, obs, prev_obs):
        raise NotImplementedError("Using virtual class " + str(self.__class__.__name__))

class TrainSubtaskController(SubtaskController):
    def __init__(self) -> None:
        pass

    def finished(self, subtask_generator, obs, prev_obs):
        if obs['inventary'] != prev_obs['inventory']:
            return True
        return False

#---------------------------------------------

if __name__ == '__main__':
    gen = RandomTargetGenerator(None, 0.1)
    print(gen.get_target(None))

class EpisodeController(gym.Wrapper):
    def __init__(self, env, target_generator, subtask_generator, task_controller, subtask_controller):
        super().__init__(env)
        self.target_generator = target_generator
        self.subtask_generator = subtask_generator
        self.task_controller = task_controller
        self.subtask_controller = subtask_controller

        self.prev_obs = None
        self.target = None
        pass

    def reset(self):
        obs = super().reset()
        self.target = self.target_generator.get_target(obs)
        self.subtask_generator.set_new_task(self.target)
        self.env.task = Task("", self.subtask_generator.get_next_subtask())
        self.prev_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.task_controller.finished(self.subtask_generator, obs, self.prev_obs):
            done = True
        if self.subtask_controller.finished(self.subtask_generator, obs, self.prev_obs):
            self.env.task = Task("", self.subtask_generator.get_next_subtask())
        self.prev_obs = obs
        return obs, reward, done, info

    def observation(self, obs, reward=None, done=None, info=None):
        obs['target_grid'] = self.env.task.target_grid
        return obs, reward, done, info