import gym
from gridworld.tasks import Task
from gridworld import GridWorld
import numpy as np
from queue import Queue

import matplotlib.pyplot as plt
np.set_printoptions(threshold=100000)

def make_iglu(*args, **kwargs):
    custom_grid = np.ones((9, 11, 11))
    env = GridWorld(render=False, select_and_place=True, discretize=True, max_steps=1000)
    env.set_task(Task("", custom_grid, invariant=False))
    tg = RandomTargetGenerator(None, 0.1)
    sg = FlyingSubtaskGenerator()
    tc = TrainTaskController()
    sc = TrainSubtaskController()
    ec = EpisodeController(env, tg, sg, tc, sc)

    return env

class TargetGenerator:
    def __init__(self, config):
        self.grid = None
        pass
    
    def get_target(self, obs):
        raise NotImplementedError("Using virtual class " + str(self.__class__.__name__))

    def plot_grid(self, text=None):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        voxel = self.grid
        idx2color = {1: 'r', 2: 'c', 3: 'y', 4: 'g', 5: 'b', 6: 'm'}
        vox = voxel.transpose(1, 2, 0)
        colors = np.empty(vox.shape, dtype=str)
        for i in range(vox.shape[0]):
            for j in range(vox.shape[1]):
                for k in range(vox.shape[2]):
                    if vox[i, j, k] != 0:
                        colors[i, j, k] = idx2color[vox[i, j, k]]

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(vox, facecolors=colors, edgecolor='k', )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        ax.set_xticks(np.arange(0, 12, 1), minor=True)
        ax.set_yticks(np.arange(0, 12, 1), minor=True)
        ax.set_zticks(np.arange(0, 9, 1), minor=True)
        if text is not None:
            plt.annotate(text, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
        plt.show()

class RandomTargetGenerator(TargetGenerator):
    def __init__(self, config, p):
        super().__init__(config)
        self.p = p
        pass

    def get_target(self, obs):
        #self.grid = np.random.choice(np.arange(0, 7), p = [1 - self.p] + [self.p / 6] * 6, size = (9, 11, 11))
        coords = np.random.normal((5,5), 1.5, size=(10, 2)).astype(int)
        coords[np.where(coords > 10)] = 10
        coords[np.where(coords < 0)] = 0
        fig = np.zeros((9,11,11))
        fig[0,coords[:,0],coords[:,1]] = 1
        self.grid = fig

        #self.grid = np.zeros((9, 11, 11))
        #self.grid[0][7][6] = 1
        #self.grid[0][8][9] = 1
        #self.grid[0][8][10] = 1
        #self.grid[0][10][10] = 1
        return self.grid

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
        if subtask_generator.empty():
            return True
        modified = (obs['inventory'] != prev_obs['inventory']).any()
        modification = obs['grid'] - prev_obs['grid']
        if modification.sum() < 0:
            modification[np.nonzero(modification)] = -1
        if modified and (modification != subtask_generator.current_task).any():
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
        if prev_obs is None:
            return False
        if (obs['grid'] != prev_obs['grid']).any():
            return True
        return False

#----------------------------------------------

class SubtaskGenerator():
    def __init__(self) -> None:
        pass

    def set_new_task(self, target):
        pass

    def get_next_subtask(self):
        pass

    def empty(self):
        pass

class FlyingSubtaskGenerator(SubtaskGenerator):
    def __init__(self):
        self.subtasks = None
        self.current_task = None
        self.prev_task = None
        pass

    def set_new_task(self, target_grid):
        self.subtasks = Queue()
        where = np.nonzero(np.sum(target_grid, axis=0))
        for i in range(len(where[0])):
            height = self.get_height(target_grid, where[0][i], where[1][i])
            for j in range(height):
                x, y = where[0][i], where[1][i]
                if target_grid[j, x, y] == 0:
                    to_put = 1
                else:
                    to_put = target_grid[j, x, y]
                target = (x, y, j, to_put)
                self.subtasks.put(target)
            holes = self.get_holes(target_grid, where[0][i], where[1][i], height)
            for hole in holes:
                target = (where[0][i], where[1][i], hole, -1)
                self.subtasks.put(target)
        pass

    def get_height(self, target_grid, x, y):
        for i in range(9):
            if target_grid[9 - i - 1][x][y] != 0:
                return 9 - i
    
    def get_holes(self, target_grid, x, y, height):
        holes = []
        for i in range(height):
            if target_grid[i][x][y] == 0:
                holes.append(i)
        return holes

    def get_next_subtask(self):
        target = self.subtasks.get()
        grid = np.zeros((9, 11, 11))
        grid[target[2], target[0], target[1]] = target[3]
        #print(grid.sum(), '--------------')
        if not (self.current_task is None):
            self.prev_task = self.current_task.copy()
        self.current_task = grid
        return grid

    def empty(self):
        return self.subtasks.empty()
#---------------------------------------------

if __name__ == '__main__':
    tc = TrainTaskController()
    tg = RandomTargetGenerator(None, 0.01)
    sg = FlyingSubtaskGenerator()
    target = tg.get_target(None)
    tg.plot_grid()
    plt.savefig('aaa')
    #sg.set_new_task(target)
    #print(target)
    #while not sg.empty():
    #    zxy = np.nonzero(sg.get_next_subtask())
        #print(zxy, end=' ')
        #print(target[zxy])
    pass

class EpisodeController(gym.Wrapper):
    def __init__(self, env, target_generator, subtask_generator, task_controller, subtask_controller):
        super().__init__(env)
        self.target_generator = target_generator
        self.subtask_generator = subtask_generator
        self.task_controller = task_controller
        self.subtask_controller = subtask_controller

        self.prev_obs = None
        self.target = None
        self.current_grid = np.zeros((9, 11, 11))
        pass

    def reset(self):
        obs = super().reset()
        self.target = self.target_generator.get_target(obs)
        self.subtask_generator.set_new_task(self.target)
        #self.current_grid = np.zeros((9, 11, 11))
        self.env.initialize_world(list(zip([], [], [], [])), (2, 2, 2, 2, 2))
        task = self.subtask_generator.get_next_subtask()
        self.env.set_task(Task("", task, invariant=False))
        self.prev_obs = obs
        self.current_grid = np.zeros((9, 11, 11))
        self.steps = 0
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        #print(self.steps, '%%%%%%%%%%%%%%%%')
    #print('----------')
        #print(obs['grid'][0])
        #print('--------')
        #print(self.prev_obs['grid'][0])
        #print(self.subtask_generator.current_task[0])
        if self.task_controller.finished(self.subtask_generator, obs, self.prev_obs):

            done = True
            print('======================')
        elif self.subtask_controller.finished(self.subtask_generator, obs, self.prev_obs):
            #self.current_grid += self.subtask_generator.current_task
            self.current_grid += self.subtask_generator.current_task
            #self.env.set_task(Task("", self.subtask_generator.get_next_subtask(), starting_grid=obs['grid']))
            #print(self.env.task)
            self.env._task = Task("", self.subtask_generator.get_next_subtask())
            print('+++++++++++++++')
        self.prev_obs = obs
        self.steps += 1
        return obs, reward, done, info
