from gridworld.tasks import Task
import numpy as np

def get_metrics(target_grid, grid, start_grid=np.zeros((9, 11, 11))):
    task = Task('', target_grid=target_grid - start_grid)

    argmax = task.argmax_intersection(grid)
    builded = grid - start_grid
    maximal_intersection = task.get_intersection(builded, *argmax)

    target_size = task.target_size
    precision = maximal_intersection / (target_size + 1e-10)
    recall = maximal_intersection / (len(builded.nonzero()[0]) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return maximal_intersection, f1

if __name__ == '__main__':
    target = np.zeros((9, 11, 11))
    grid = np.zeros((9, 11, 11))

    target[0, 0, 0] = 1
    target[1, 1, 0] = 1
    grid[0, 0, 0] = 1
    grid[1, 0, 1] = 1

    print(get_metrics(target, grid))