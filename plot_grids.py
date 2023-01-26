import numpy as np

targets = np.load('targets.npy')
preds = np.load('preds.npy')
builts = np.load('builts.npy')

def plot_grid(voxels, text=None, filename=None, ax=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(40, 40))

    for iii, voxelll in enumerate(voxels):
        for jjj, voxel in enumerate(voxelll):
            if jjj > 16:
                break
            idx2color = {1: 'r', 2: 'c', 3: 'y', 4: 'g', 5: 'b', 6: 'm'}
            vox = voxel.transpose(1, 2, 0)
            colors = np.empty(vox.shape, dtype=str)
            for i in range(vox.shape[0]):
                for j in range(vox.shape[1]):
                    for k in range(vox.shape[2]):
                        if vox[i, j, k] != 0:
                            colors[i, j, k] = idx2color[vox[i, j, k]]
                            
            ax = fig.add_subplot(17, 3, jjj*3 + iii + 1, projection='3d')
            ax.voxels(vox, facecolors=colors, edgecolor='k', )

            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
            ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
            ax.set_xticks(np.arange(0, 12, 1), minor=True)
            ax.set_yticks(np.arange(0, 12, 1), minor=True)
            ax.set_zticks(np.arange(0, 9, 1), minor=True)
    fig.subplots_adjust(wspace=0, hspace=0)
    if text is not None:
        ax.annotate(text, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    filename='buf.png'
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

i = 11
plot_grid([targets[np.array([1,  8,  9, 11, 12, 13, 14, 15, 19, 20, 30, 31, 32, 33, 34, 35, 36, 37]) - 1],
preds[np.array([1,  8,  9, 11, 12, 13, 14, 15, 19, 20, 30, 31, 32, 33, 34, 35, 36, 37]) - 1],
builts[np.array([1,  8,  9, 11, 12, 13, 14, 15, 19, 20, 30, 31, 32, 33, 34, 35, 36, 37]) - 1]])
plt.show()

print(targets.shape)