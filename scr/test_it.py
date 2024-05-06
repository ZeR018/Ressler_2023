import numpy as np
from matplotlib import pyplot as plt
from config import settings as s
import memory_worker as mem

# # Пытаемся расположить графики
# gs_kw = dict(width_ratios=[1.5, 1], height_ratios=[1,1,1,1,1,1])
# fig, axd = plt.subplot_mosaic([['xt', 'xyz',],
#                                ['xt', 'xyz',],
#                                ['yt', 'xyz',],
#                                ['yt', 'xy',],
#                                ['zt', 'xy',],
#                                ['zt', 'xy',]],
#                               gridspec_kw=gs_kw, figsize=(12, 8),
#                               layout="constrained")
# for k in axd:
#     axd[k].plot(np.random.random(10))
# fig.suptitle('plt.subplot_mosaic()')

# plt.show()


# fig = plt.figure(figsize=(10, 6))  

# axis_1 = fig.add_subplot(2, 2, 2, projection='3d')
# axis_1_track = fig.add_subplot(2, 2, 4)
# axis_2_x = fig.add_subplot(4, 2, 1)
# axis_2_y = fig.add_subplot(4, 2, 3)
# axis_2_z = fig.add_subplot(4, 2, 5)
# axis_3 = fig.add_subplot(4, 2, 7)
# line, = axis_1.plot3D(np.random.random(), np.random.random(), np.random.random())
# axis_1_track = plt.plot(np.random.random(), np.random.random())
p_name = 'a0.22'
times_of_sync = mem.read_times_series_experiments(f'./data/grid_experiments/series100t200/{p_name}/times.txt')

plt.figure()
plt.hist(times_of_sync, 20)
plt.grid()
plt.xlabel('Время синхронизации')
plt.ylabel('Число синхронизаций')
plt.savefig(f'./data/grid_experiments/series100t200/{p_name}/times_hist2.png')

plt.show()