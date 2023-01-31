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
        return False

import matplotlib.pyplot as plt
np.set_printoptions(threshold=100000)

def make_figlu(*args, **kwargs):
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
    
    def get_target(self):
        raise NotImplementedError("Using virtual class " + str(self.__class__.__name__))

def plot_grid(voxel, text=None, filename=None, ax=None):
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
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

class RandomTargetGenerator(TargetGenerator):
    def __init__(self, config, p):
        super().__init__(config)
        self.p = p
        pass

    def get_target(self):
        #self.grid = np.random.choice(np.arange(0, 7), p = [1 - self.p] + [self.p / 6] * 6, size = (9, 11, 11))
        #self.grid[4, 4, 4] = 1
        
        #coords = np.random.normal((5,5), 1.5, size=(10, 2)).astype(int)
        #coords[np.where(coords > 10)] = 10
        #coords[np.where(coords < 0)] = 0
        #fig = np.zeros((9,11,11))
        #fig[0,coords[:,0],coords[:,1]] = 1
        #self.grid = fig
        #for xyz in np.transpose(np.where(fig)):
        #    h = np.random.choice(np.arange(1, 9))
        #    for i in range(h):
        #        if np.random.choice([0, 1], p=[0.5, 0.5]):
        #            fig[i, xyz[1], xyz[2]] = 1
        #self.grid = fig

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

        last_color = np.random.choice(np.arange(1, 7))

        r = np.random.normal(5.5, 1, (10, 2))
        r = r.astype(int)
        r[r < 0] = 0
        r[r > 10] = 10
        r = np.unique(r, axis=0)

        self.grid = np.zeros((9, 11, 11))
        for idx in r:
            for h in range(4):
                put = np.random.choice([0, 1], p=[0.7, 0.3])
                if put:
                    colors_prob = [1/10] * 6
                    colors_prob[last_color - 1] = 1/2
                    color = np.random.choice(np.arange(1, 7), p=colors_prob)
                    last_color = color
                    self.grid[h, idx[0], idx[1]] = color

        if self.grid.sum() == 0:
            x = np.random.choice(np.arange(0, 10))
            y = np.random.choice(np.arange(0, 10))
            self.grid[0][x][y] = np.random.choice(np.arange(1, 7))

        #self.grid = np.zeros((9, 11, 11))
        #x = np.random.choice(np.arange(11))
        #y = np.random.choice(np.arange(11))
        #z = np.random.choice(np.arange(1, 4))
        #self.grid[0, x, y] = 1


        return self.grid

#---------------------------------------------

class TaskController():
    def __init__(self) -> None:
        pass

    def finished(self, obs, prev_obs, task):
        raise NotImplementedError("Using virtual class " + str(self.__class__.__name__))

class TrainTaskController(TaskController):
    def __init__(self) -> None:
        pass

    def finished(self, obs, prev_obs, task):
        modified = (obs['grid'] != prev_obs['grid']).any()
        modification = obs['grid'] - prev_obs['grid']
        modification[modification > 0] = 1
        modification[modification < 0] = -1
        if modified and (modification != task).any():
            return True
        return False

class TestTaskController(TaskController):
    def __init__(self):
        pass

    def finished(self, obs, prev_obs, task):
        return False

#---------------------------------------------

class SubtaskController():
    def __init__(self) -> None:
        pass

    def finished(self, obs, prev_obs, task):
        raise NotImplementedError("Using virtual class " + str(self.__class__.__name__))

class TrainSubtaskController(SubtaskController):
    def __init__(self) -> None:
        pass

    def finished(self, obs, prev_obs, task):
        if prev_obs is None:
            return False
        if (obs['grid'] != prev_obs['grid']).any():
            return True
        return False

#----------------------------------------------

class SubtaskGenerator():
    def __init__(self) -> None:
        pass

    def set_new_task(self, target, start_grid):
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

    def set_new_task(self, target_grid, start_grid):
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
        if self.subtasks.empty():
            return None
        target = self.subtasks.get()
        grid = np.zeros((9, 11, 11))
        grid[target[2], target[0], target[1]] = target[3]
        #if not (self.current_task is None):
        #    self.prev_task = self.current_task.copy()
        #self.current_task = grid
        return grid

    def empty(self):
        return self.subtasks.empty()

class ggNavigationSubtaskGenerator(SubtaskGenerator):
    def __init__(self):
        self.subtasks = None
        self.prev_task = None
        self.current_task = None
        self.put_prob = 1.0

    def set_new_task(self, target):
        from copy import deepcopy
        self.target = target
        self.subtasks = Queue()

        if np.random.choice([0, 1], p=[1-self.put_prob, self.put_prob]):
            free = set()
            where = list(np.where(target))
            for i in range(11):
                for j in range(11):
                    if np.random.choice([0, 1], p=[0.9, 0.1]):
                        free.add((i, j, 0))
            for i in range(3):
                neighbs = deepcopy(where)
                neighbs[i] += 1
                for j in range(len(neighbs[0])):
                    targ = (neighbs[1][j], neighbs[2][j], neighbs[0][j])
                    if targ[0] >=0 and targ[0] < 11:
                        if targ[1] >=0 and targ[1] < 11:
                            if targ[2] >=0 and targ[2] < 9:
                                free.add(targ)

                neighbs[i] -= 2
                for j in range(len(neighbs[0])):
                    targ = (neighbs[1][j], neighbs[2][j], neighbs[0][j])
                    if targ[0] >=0 and targ[0] < 11:
                        if targ[1] >=0 and targ[1] < 11:
                            if targ[2] >=0 and targ[2] < 9:
                                free.add(targ)

            for j in range(len(neighbs[0])):
                free.discard((where[1][j], where[2][j], where[0][j]))
            n = len(free)
            i = np.random.choice(np.arange(n))
            free = list(free)
            targ = (free[i][0], free[i][1], free[i][2], 1)
            self.subtasks.put(targ)

        else:
            blocks = np.where(self.target)
            n = len(blocks[0])
            i = np.random.choice(np.arange(n))
            self.subtasks.put((blocks[1][i], blocks[2][i], blocks[0][i], -1))

    def get_next_subtask(self):
        target = self.subtasks.get()
        grid = np.zeros((9, 11, 11))
        grid[target[2], target[0], target[1]] = target[3]
        if not (self.current_task is None):
            self.prev_task = self.current_task.copy()
        self.current_task = grid
        self.prev_task = self.current_task.copy()
        return grid

    def empty(self):
        return self.subtasks.empty()

class NavigationWorldInitializer:
    def __init__(self):
        self.put_prob = 0.5
        pass

    def init_world(self, target):
        start_grid = np.zeros((9, 11, 11))

        blocks = np.transpose(np.nonzero(target))
        n = blocks.shape[0]

        subtask = np.random.choice(np.arange(n))
        subtask = blocks[subtask]

        for x in range(subtask[1]):
            for y in range(11):
                for z in range(9):
                    start_grid[z, x, y] = target[z, x, y]
        x = subtask[1]
        for y in range(subtask[2]):
            for z in range(9):
                start_grid[z, x, y] = target[z, x, y]
        y = subtask[2]
        for z in range(subtask[0]):
            start_grid[z, x, y] = target[z, x, y]
        
        p = self.put_prob
        if np.random.choice([1, 0], p=[p, 1-p]):
            if not self._has_adj(subtask, start_grid):
                for z in range(subtask[0]):
                    start_grid[z, x, y] = 1
        
            new_target = start_grid.copy()
            new_target[subtask[0], x, y] = target[subtask[0], x, y]
        else:
            h = np.random.choice(np.arange(1, 8))
            for z in range(h):
                start_grid[z, x, y] = 1
            
            for_del = np.random.choice(np.arange(h))

            new_target = start_grid.copy()
            new_target[for_del, x, y] = 0

        x = np.random.choice(np.arange(11))
        y = np.random.choice(np.arange(11))
        z = np.random.choice(np.arange(6))
        yaw = np.random.choice(np.arange(-175, 175, 5))
        pitch = np.random.choice(np.arange(-90, 25, 5))

        while start_grid[z, x, y] != 0 and start_grid[z + 1, x, y] != 0:
            x = np.random.choice(np.arange(11))
            y = np.random.choice(np.arange(11))
            z = np.random.choice(np.arange(6))

        pos = (x - 5, z, y - 5, yaw, pitch)

        return start_grid, pos, new_target

    def visualize(self, target, filename):
        start_grid, pos, target = self.init_world(target)

        start_grid[np.nonzero(start_grid)] = 6
        target[np.nonzero(target)] = 6

        subtask_grid = target - start_grid
        subtask = tuple(np.transpose(np.nonzero(subtask_grid))[0].tolist())
        x, z, y, _, _ = pos
        x += 5
        y += 5
        if (z, x, y) == subtask:
            start_grid[z, x, y] = 3
        else:
            if subtask_grid.sum() > 0:
                start_grid[subtask] = 4
            else:
                start_grid[subtask] = 1
            start_grid[z, x, y] = 2

        plot_grid(start_grid, filename=filename)

    
    def _has_adj(self, subtask, grid):
        if subtask[0] == 0:
            return True
        neibs = [[-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]]

        for dv in neibs:
            neib = subtask + dv
            if (neib >= 0).all() and (neib[1:] < 11).all() and neib[0] < 9:
                if grid[tuple(neib)] != 0:
                    return True
        return False

class NavigationSubtaskGenerator(SubtaskGenerator):
    def __init__(self):
        self.subtasks = None

    def set_new_task(self, target, start_grid):
        subtask_grid = target - start_grid
        subtask_grid[subtask_grid < 0] = -1
        subtask = np.transpose(np.nonzero(subtask_grid))[0].tolist()
        self.subtasks = Queue()
        self.subtasks.put(tuple(subtask + [subtask_grid[tuple(subtask)]]))

    def get_next_subtask(self):
        if self.subtasks.empty():
            return None
        target = self.subtasks.get()
        grid = np.zeros((9, 11, 11))
        grid[target[0], target[1], target[2]] = target[3]
        return grid

class VisualNavigationWorldInitializer():
    def __init__():
        pass

    def init_world(self, target):
        start_grid = np.zeros((9, 11, 11))
        x = np.random.choice(np.arange(5))
        y = np.random.choice(np.arange(1, 11))
        z = np.random.choice(np.arange(6))

        start_grid[z, x, y] = 1
        target = start_grid.copy()
        target[z, x, y - 1] = 1

        x = np.random.choice(np.arange(7, 11))
        y = np.random.choice(np.arange(1, 11))
        z = np.random.choice(np.arange(6))

        pos = (x - 5, z, y - 5, 0, 0)

        return start_grid, pos, target

#---------------------------------------------

if __name__ == '__main__':
    a = NavigationWorldInitializer()
    for i in range(10):
        t = RandomTargetGenerator(None, None).get_target()
        a.visualize(t, 'nav/nav' + str(i) + '.png')

def startgridformat(start_grid):
    blocks = np.where(start_grid)
    ind = np.lexsort((blocks[0], blocks[2], blocks[1]))
    Zorig, Xorig, Yorig = blocks[0][ind] - 1, blocks[1][ind] - 5, blocks[2][ind] - 5
    ids = []
    for i in range(len(Zorig)):
        ids.append(int(start_grid[Zorig[i] + 1, Xorig[i] + 5, Yorig[i] + 5]))
    starting_grid = list(zip(Xorig, Zorig, Yorig, ids))
    return starting_grid

class EpisodeController(gym.Wrapper):
    def __init__(self, env, target_generator, subtask_generator, task_controller, subtask_controller, world_initializer, max_subtask_step=350):
        super().__init__(env)
        self.target_generator = target_generator
        self.subtask_generator = subtask_generator
        self.task_controller = task_controller
        self.subtask_controller = subtask_controller
        self.world_initializer = world_initializer

        self.max_subtask_step = max_subtask_step
        self.target = None
        self.subtask_step = 0
        self.steps = 0
        pass

    def reset(self):
        obs = super().reset()
        self.subtask_step = 0
        self.steps = 0
        self.target = self.target_generator.get_target()
        start_grid, start_pos, new_target = self.world_initializer.init_world(self.target)
        if not new_target is None:
            self.target = new_target
        self.subtask_generator.set_new_task(self.target, start_grid)

        obs['grid'] = start_grid
        self.env.set_prev_obs(obs)

        starting_grid = startgridformat(start_grid)

        self.env.initialize_world(starting_grid, start_pos)
        subtask = self.subtask_generator.get_next_subtask()
        self.env.set_task(Task("", self.target, invariant=False))
        self.env.set_subtask(Task("", subtask, invariant=False))
        self.prev_task = self.env.subtask
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        self.steps += 1
        self.subtask_step += 1

        task = self.env.subtask.target_grid
        if self.task_controller.finished(obs, self.prev_obs, task) or self.subtask_step > self.max_subtask_step:
            if self.subtask_step > self.max_subtask_step:
                reward -= 1.0
            done = True
            print(1, '{{{{{{{{{{{{')
            print(self.prev_obs['grid'].nonzero())
            print(obs['grid'].nonzero())
            print(task.nonzero())
        elif self.subtask_controller.finished(obs, self.prev_obs, task):
            subtask = self.subtask_generator.get_next_subtask()
            if subtask is None:
                print(2, '{{{{{{{{{{{{{{{{{{')
                done = True
            else:
                #self.env.set_task(Task("", subtask, invariant=False))
                self.prev_task = self.env.subtask
                self.env.set_subtask(Task("", subtask, invariant=False))
            self.subtask_step = 0
        if done:
            print('(((((((((((((((((')

        return obs, reward, done, info
