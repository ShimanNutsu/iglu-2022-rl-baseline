import numpy as np

from wrappers.common_wrappers import Wrapper

import gym


def strict_reward_range():
    reward_range = [1, 0.25, 0.05, 0.001, -0.0001, -0.001, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08,
                    -0.09]
    for_long_distance = [-0.10 - 0.01 * i for i in range(4)]
    return reward_range + for_long_distance


def remove_reward_range():
    reward_range = [1, 0.0001, 0.00, 0.00, -0.0001, -0.001, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08,
                    -0.09]
    for_long_distance = [-0.10 - 0.01 * i for i in range(4)]
    return reward_range + for_long_distance


class OneBlockReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = 0
        modification = obs['grid'] - self.prev_obs['grid']
        if modification.sum() == 0:
            return obs, reward, done, info
        
        action_type = 1 if modification.sum() > 0 else -1
        task_type = 1 if self.env.task.target_grid.sum() > 0 else -1
        if action_type != task_type:
            reward += -0.001
        else:
            block = obs['grid'] - self.prev_obs['grid']
            block[np.nonzero(block)] = 1 if block.sum() > 0 else -1
            target_block = self.env.task.target_grid

            coords1 = np.transpose(np.nonzero(block))[0]
            coords2 = np.transpose(np.nonzero(target_block))[0]
            dist = np.linalg.norm(coords1 - coords2)

            if task_type == 1:
                reward += strict_reward_range()[int(dist)]
            else:
                reward += remove_reward_range()[int(dist)]
        return obs, reward, done, info

class PutUnderReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        modification = obs['grid'] - self.prev_obs['grid']
        if modification.sum() > 0:
            block = obs['grid'] - self.prev_obs['grid']
            block[np.nonzero(block)] = 1 if block.sum() > 0 else -1
            target_block = self.env.task.target_grid

            coords1 = np.transpose(np.nonzero(block))[0]
            coords2 = np.transpose(np.nonzero(target_block))[0]
            dist = np.linalg.norm(coords1 - coords2)

            x_agent, z_agent, y_agent = obs['agentPos'][:3]
            x_agent, y_agent = x_agent + 5, y_agent + 5
            x_agent, y_agent = int(x_agent + 0.5), int(y_agent + 0.5)

            if int(dist) == 0:
                if coords1[1] == x_agent and coords1[2] == y_agent and (z_agent - coords1[0]) >= 0:
                    reward += 0.5
        return obs, reward, done, info



class RangetReward(Wrapper):
    def __init__(self, env, rspec=15):
        super().__init__(env)
        self.rspec = rspec

    def calc_reward(self, dist, remove=False):
        reward_range = strict_reward_range()
        remove_reward_range_ = remove_reward_range()
        try:
            if remove:
                reward = remove_reward_range_[int(dist)]
            else:
                reward = reward_range[int(dist)]
        except Exception as e:
            raise Exception(e)
        return reward

    def blocks_count(self, info):
        return np.sum(info['grid'] != 0)

    def check_goal_closeness(self, info=None, broi=None, remove=False):
        roi = np.where(self.env.subtask_generator.prev_task != 0)
        goal = np.mean(roi[1]), np.mean(roi[2]), np.mean(roi[0])
        print('::::::::::::::::::::', goal)
        if broi is None:
            broi = np.where(info['grid'] != 0)  # y x z
        builds = np.mean(broi[1]), np.mean(broi[2]), np.mean(broi[0])
        print('::::::::::::::::::::', builds)
        dist = ((goal[0] - builds[0]) ** 2 +
                (goal[1] - builds[1]) ** 2 +
                (goal[2] - builds[2]) ** 2) ** 0.5
        return self.calc_reward(dist, remove)


def calc_new_blocks(current_grid, last_grid):
    grid = np.zeros_like(current_grid)
    relief = np.zeros_like(last_grid)
    grid[current_grid != 0] = 1
    relief[last_grid != 0] = 1

    new_blocks = np.where(grid != relief)
    if len(new_blocks[0]) > 1:
        np.set_printoptions(threshold=100000)
        raise Exception(f"""
               Bulded more then one block! Logical error!!
               grid z_x_y- {np.where(current_grid != 0)}
               relief z_x_y- {np.where(last_grid != 0)}
               blocks z_x_y- {np.where(grid != relief)}
               """)

    return grid, relief, np.where(grid != relief)


class RangetRewardFilledField(RangetReward):
    def __init__(self, env):
        super().__init__(env)
        self.fs = True
        self.info = dict()
        self.last_obs = None
        self.last_grid = None

    def reset(self):
        self.fs = True
        self.SR = 0
        self.steps = 0
        self.tasks_count = 1
        self.last_grid = None
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.last_grid is None:
            self.last_grid = obs['grid']
            info['done'] = 'len_done_%s' % self.steps
            return obs, reward, done, info

        if done:
            info['done'] = 'len_done_%s' % self.steps
        info['done'] = 'len_done_%s' % self.steps
        self.last_obs = obs
        self.steps += 1

        ### Calc count of new blocks
        grid, relief, new_blocks = calc_new_blocks(obs['grid'], self.last_grid)
        info['done_grid'] = grid
        info['episode_extra_stats'] = info.get('episode_extra_stats', {})

        ### Reward calculation
        reward = 0
        if not done:
            current_task = self.env.subtask_generator.prev_task
        else:
            current_task = self.env.subtask_generator.current_task
        task = np.sum(current_task)  # if < 0 - task if remove, else task is build
        if len(new_blocks[0]) >= 1:
            grid_block_count = len(np.where(grid != 0)[0])
            relief_block_count = len(np.where(relief != 0)[0])

            if task < 0 and grid_block_count > relief_block_count:  # если нужно удалить кубик, а агент его поставил
                reward = -0.001
            elif task > 0 and grid_block_count < relief_block_count:  # если нужно поставить кубик, а агент его удалил
                reward = -0.001
            else:
                reward = self.check_goal_closeness(info, broi=new_blocks, remove=task < 0)  # иначе

            if task < 0:
                do = 0
            elif task > 0:
                do = int(np.sum(current_task))

            ### Add reward for block under agent
            x_agent, z_agent, y_agent = obs['agentPos'][:3]
            x_agent, y_agent = x_agent + 5, y_agent + 5
            x_agent, y_agent = int(x_agent + 0.5), int(y_agent + 0.5)
            z_last_block, x_last_block, y_last_blcok = np.where(current_task != 0)
            if reward == 1:
                self.SR += 1
                if x_last_block == x_agent and y_last_blcok == y_agent and (z_agent - z_last_block) <= 2:
                    reward += 0.5
                if task < 0:
                    if int(x_last_block - x_agent) >= 0 and int(
                            y_last_blcok - y_agent) >= 0 and z_agent >= z_last_block:
                        #     raise Exception("WRONG!")
                        reward += 0.5
                #full = self.env.one_round_reset(new_blocks, do)
                full = self.env.task_controller.finished(self.env.subtask_generator, obs, self.env.prev_obs)
                info['done'] = 'right_move'
                if full:
                    info['done'] = 'full'
                    #done = True

            if reward < 1:
                info['done'] = 'mistake_%s' % self.steps
                #done = True
                #self.env.update_field(new_blocks, do)
            if done:
                info['episode_extra_stats']['SuccessRate'] = self.SR / self.tasks_count
            self.tasks_count += 1
        self.last_grid = obs['grid']
        self.fs = False
        self.info = info
        return obs, reward, done, info


class Closeness(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dist = 1000000

    def reset(self):
        self.dist = 1000000
        return super().reset()

    def closeness(self, info):

        roi = np.where(self.env.task.target_grid != 0)  # y x z
        goal = np.mean(roi[1]), np.mean(roi[2]), np.mean(roi[0])
        agent = info['agentPos'][:3]
        agent_pos = agent[0] + 5, agent[2] + 5, agent[1] + 1

        dist = ((goal[0] - agent_pos[0]) ** 2 + (goal[1] - agent_pos[1]) ** 2 + (goal[2] - agent_pos[2]) ** 2) ** 0.5
        return dist

    def calc_reward(self, info):
        d2 = self.closeness(info)
        if d2 < self.dist:
            self.dist = d2
            return 0.001
        else:
            return 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        add_reward = self.calc_reward(obs)
        reward += add_reward
        return obs, reward, done, info
