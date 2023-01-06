import gym
from gridworld.tasks import Task
from gridworld import GridWorld
import numpy as np
from queue import Queue
from collections import deque

import gc

class Q():
    def __init__(self):
        self.queue = deque()
        pass
    def put(self, val):
        self.queue.append(val)

    def get(self):
        return self.queue.popleft()
    def empty(self):
        if self.queue:
            return True
        print('-')
        return False

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

def plot_grid(voxel, text=None, file=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

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
    if file is None:
        plt.show()
    else:
        plt.savefig(file)

class RandomTargetGenerator(TargetGenerator):
    def __init__(self, config, p):
        super().__init__(config)
        self.p = p
        pass

    def get_target(self, obs):
        #self.grid = np.random.choice(np.arange(0, 7), p = [1 - self.p] + [self.p / 6] * 6, size = (9, 11, 11))
        #self.grid[4, 4, 4] = 1
        
        coords = np.random.normal((5,5), 1.5, size=(10, 2)).astype(int)
        coords[np.where(coords > 10)] = 10
        coords[np.where(coords < 0)] = 0
        fig = np.zeros((9,11,11))
        fig[0,coords[:,0],coords[:,1]] = 1
        self.grid = fig
        for xyz in np.transpose(np.where(fig)):
            h = np.random.choice(np.arange(1, 9))
            for i in range(h):
                if np.random.choice([0, 1], p=[0.5, 0.5]):
                    fig[i, xyz[1], xyz[2]] = 1
        self.grid = fig

        #self.grid = np.zeros((9, 11, 11))
        #self.grid[0][7][6] = 1
        #self.grid[0][8][9] = 1
        #self.grid[0][8][10] = 1
        #self.grid[0][10][10] = 1
        
        #self.grid = np.zeros((9, 11, 11))
        #x = np.random.choice(np.arange(0, 10))
        #y = np.random.choice(np.arange(0, 10))
        #self.grid[0][x][y] = 1
        #self.grid[8][10][10] = 1

        #r = np.random.normal(5.5, 1, (10, 2))
        #r = r.astype(int)
        #r[r < 0] = 0
        #r[r > 10] = 10
        #r = np.unique(r, axis=0)

        #self.grid = np.zeros((9, 11, 11))
        #for idx in r:
        #    for h in range(4):
        #        self.grid[h, idx[0], idx[1]] = np.random.choice([0, 1], p=[0.5, 0.5])

        #self.grid = np.zeros((9, 11, 11))
        #x = np.random.choice(np.arange(4, 9))
        #y = np.random.choice(np.arange(4, 9))
        #z = np.random.choice(np.arange(1, 4))
        #self.grid[z, x, y] = 1


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
        modified = (obs['grid'] != prev_obs['grid']).any()
        modification = obs['grid'] - prev_obs['grid']
        modification[modification > 0] = 1
        modification[modification < 0] = -1
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

    def test(self, target, metric):
        self.set_new_task(target)
        grid = np.zeros((9, 11, 11))
        while not self.empty():
            s0 = self.get_next_subtask()
            s = np.where(s0)
            grid[s[0], s[1], s[2]] += s0[s[0], s[1], s[2]]
        return metric(target, grid)

    def visualize(self, target, file):
        from PIL import Image
        import os

        plot_grid(target, file='targ.png')

        targ = Image.open('targ.png').resize((300, 200))
        self.set_new_task(a)
        grid = np.zeros((9, 11, 11))

        import cv2
        videodims = (640, 480)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
        video = cv2.VideoWriter(file + ".mp4", fourcc, 2, videodims)

        while not self.empty():
            s0 = self.get_next_subtask()
            s = np.where(s0)
            grid[s[0], s[1], s[2]] += s0[s[0], s[1], s[2]]
            plot_grid(grid, file='buf.png')
            step = Image.open('buf.png')
            step.paste(targ, (0, 0))
            video.write(cv2.cvtColor(np.array(step), cv2.COLOR_RGB2BGR))
        os.remove("buf.png")
        os.remove("targ.png")

class WalkingSubtaskGenerator(SubtaskGenerator):
    def __init__(self):
        self.subtasks = Queue()
        self.current_task = None
        self.prev_task = None
        pass

    def set_new_task(self, target_grid):
        self.subtasks = Queue()
        where = np.nonzero(np.sum(target_grid, axis=0))
        for i in range(len(where[0])):
            height = self.get_height(target_grid, where[0][i], where[1][i])
            x, y = where[0][i], where[1][i]
            for j in range(height):
                if target_grid[j, x, y] == 0:
                    to_put = 1
                else:
                    to_put = target_grid[j, x, y]
                target = (x, y, j, to_put)
                self.subtasks.put(target)
            holes = self.get_holes(target_grid, where[0][i], where[1][i], height)
            
            try:
                if x < 10 and y < 10:
                    for aux in range(holes[-1]):
                        target = (x + 2, y + 2, aux, 1)
                        self.subtasks.put(target)
                    cur = holes[-1]
                    for hole in reversed(holes):
                        while cur != hole:
                            target = (x + 2, y + 2, cur - 1, -1)
                            self.subtasks.put(target)
                            cur -= 1
                        target = (x, y, hole, -1)
                        self.subtasks.put(target)
                    for i in range(cur, 0, -1):
                        target = (x + 2, y + 2, i - 1, -1)
                        self.subtasks.put(target)
            except:
                pass
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
        if not (self.current_task is None):
            self.prev_task = self.current_task.copy()
        self.current_task = grid
        return grid

    def empty(self):
        return self.subtasks.empty()

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
            for hole in reversed(holes):
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
        if not (self.current_task is None):
            self.prev_task = self.current_task.copy()
        self.current_task = grid
        return grid

    def empty(self):
        return self.subtasks.empty()
#---------------------------------------------

if __name__ == '__main__':
    gr = RandomTargetGenerator(None, 0.01)
    a = gr.get_target(None)
    st = WalkingSubtaskGenerator()
    st.visualize(a, 'walking')


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

    def _get_subgrid(self, grid):
        sz = int(len(np.where(grid)[0])*0.6) - 1
        if sz < 3:
            n = 0
        else:
            n = np.random.choice(list(range(sz)), p=[0.7] + [0.3 / (sz-1) for _ in range(sz-1)])
        n = 0
        subgrid = np.zeros((9, 11, 11))

        whr = np.transpose(np.where(grid))
        for i, xyz in enumerate(whr):
            if i == n:
                break
            subgrid[xyz[0], xyz[1], xyz[2]] = 1

        grid = grid - subgrid

        blocks = np.where(subgrid)
        ind = np.lexsort((blocks[0], blocks[2], blocks[1]))
        Zorig, Xorig, Yorig = blocks[0][ind] - 1, blocks[1][ind] - 5, blocks[2][ind] - 5
        ids = [1] * len(Zorig)
        starting_grid = list(zip(Xorig, Zorig, Yorig, ids))
        return starting_grid, grid, subgrid

    def reset(self):
        #while not self.subtask_generator.empty():
        #    self.subtask_generator.get_next_subtask()
        #import sys
        #print(sys.getrefcount(self.subtask_generator.subtasks.queue))
        #print(sys.getrefcount(self))
        #self.subtask_generator.subtasks.queue.clear()
        obs = super().reset()
        self.prev_obs = obs
        self.target = self.target_generator.get_target(obs)
        #strt, gr, self.prev_obs['grid'] = self._get_subgrid(self.target)
        #self.target = gr

        try:
            pos = (strt[-1][1], strt[-1][2], strt[-1][0] + 2, 0, 0)
        except:
            pos = (0, 0, 0, 0, 0)
        #self.env.initialize_world(strt, pos)
        self.subtask_generator.set_new_task(self.target)
        n = np.transpose(np.where(self.target)).shape[0]
        #n = 3*n
        preb = np.random.choice(np.arange(1, n))
        start_grid = np.zeros((9, 11, 11))
        last = [2, 5, 5]
        if np.random.choice([0, 1], p=[0.3, 0.7]):
            self.env.initialize_world([], (0, 0, 0, 0, 0))
        else:
            for i in range(preb):
                task = self.subtask_generator.get_next_subtask()
                start_grid += task
                if task.sum() > 0:
                    last = np.transpose(np.where(task))[0]
                last_act = task.sum()
            pos = (last[1] - 5, last[0] + 2, last[2] - 5, 0, -90 if last_act == 1 else 0)
            blocks = np.where(start_grid)
            ind = np.lexsort((blocks[0], blocks[2], blocks[1]))
            Zorig, Xorig, Yorig = blocks[0][ind] - 1, blocks[1][ind] - 5, blocks[2][ind] - 5
            ids = [1] * len(Zorig)
            starting_grid = list(zip(Xorig, Zorig, Yorig, ids))
            self.env.initialize_world(starting_grid, pos)

        task = self.subtask_generator.get_next_subtask()
        self.env.set_task(Task("", task, invariant=False))
        self.prev_obs['grid'] = start_grid
        self.current_grid = np.zeros((9, 11, 11))
        self.steps = 0
        self.sub_step = 0
        return obs

    def step(self, action):
        self.sub_step += 1
        obs, reward, done, info = super().step(action)
        if self.task_controller.finished(self.subtask_generator, obs, self.prev_obs) or self.sub_step > 150:
            if not self.subtask_generator.empty():
                self.env._task = Task("", self.subtask_generator.get_next_subtask(), invariant=False)

            done = True
            print('======================')
        elif self.subtask_controller.finished(self.subtask_generator, obs, self.prev_obs):
            self.sub_step = 0
            #self.env.set_task(Task("", self.subtask_generator.get_next_subtask(), invariant=False))
            self.env._task = Task("", self.subtask_generator.get_next_subtask(), invariant=False)
            print('+++++++++++++++')
        else:
            pass
            #print(';', end='')
        self.prev_obs = obs
        self.steps += 1
        return obs, reward, done, info
