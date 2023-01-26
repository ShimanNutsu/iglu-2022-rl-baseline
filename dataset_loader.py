import pandas as pd
import numpy as np
import re

def dataset2grids(filename):
    data = pd.read_excel('example.xlsx', header=1)

    data = data.drop(['Unnamed: 0', 'task'], axis=1)
    target = []
    chatgpt = []
    for i in range(len(data)):
        s = data['chat_gpt_grid'][i]
        if pd.isna(s):
            continue
        nums = list(map(int, re.findall(r'[+-]?[.]?[\d]+', s)))
        if nums == []:
            continue
        arr = np.array(np.reshape(nums, (-1 ,3)))
        arr[:, [0, 1, 2]] = arr[:, [2, 0, 1]]
        chatgpt.append(arr)

        s = data['target_grid'][i]
        nums = list(map(int, re.findall(r'\d+', s)))
        target.append(np.reshape(nums, (-1 ,3)))
    return target, chatgpt

def lst2grisd(lst):
    grid = np.zeros((9, 11, 11))
    try:
        add_x = 5 - int(lst[:, 1].mean())
        add_y = 5 - int(lst[:, 2].mean())
    except:
        print(lst)
    add = np.array([0, add_x, add_y])
    lst = lst + add
    for c in lst:
        try:
            grid[tuple(c)] = 1
        except:
            print(c, add)
    return grid

def lst2grid(lst):
    grid = np.zeros((9, 11, 11))
    x = [lst[:, 1].min(), lst[:, 1].max()]
    y = [lst[:, 2].min(), lst[:, 2].max()]
    z = [lst[:, 0].min(), lst[:, 0].max()]
    z[0] = min(0, z[0])
    lst = lst - np.array([z[0], x[0] + int(x[1]/2 - x[0]/2) - 5, y[0] + int(y[1]/2 - y[0]/2) - 5])
    lst[:, [0, 1]] = lst[:, [1, 0]]
    for c in lst:
        try:
            grid[tuple(c)] = 1
        except:
            print(c)
    return grid

def get_grids():
    target, pred = dataset2grids(0)

    for i in range(len(target)):
        target[i] = lst2grid(target[i])
        pred[i] = lst2grid(pred[i])
    
    return target, pred

from metrics import get_metrics

targets, preds = get_grids()

m = []
for i in range(len(targets)):
    m.append(get_metrics(targets[i], preds[i])[1])

m = np.array(m)
print(m.mean())