import memory_worker as mem
import matplotlib.pyplot as plt
import numpy as np
import settings as s


k_elements = 5
end_t = 3

# fig = plt.figure(figsize=[6.5, 3])
# plt.subplots_adjust(left=0.19, right=0.97, top=0.95, bottom=0.28)

# path = 'lorenz_5_elements_sync'
# path = '2024-12-04 14.37.49 2lorens_r100_coup_y'
# xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')
# print('Data 1 taken')

# plot_colors = mem.make_colors(k_elements)
# end = np.searchsorted(ts, end_t, side='left')
# for agent in range(k_elements):
#     plt.plot(ts[:end], xs[agent][:end], color=plot_colors[agent], label=f'agent {agent+1}')
# plt.xlim(0, end_t)
# plt.xlabel('t', fontsize=28)
# plt.ylabel('x', fontsize=28)
# plt.xticks([0,1,2,3],fontsize=24)
# plt.yticks([0, 10, -20],fontsize=24)
# plt.legend(loc='lower right')


# path = '2024-12-04 14.37.49 2lorens_r100_coup_y'
# ax2 = fig.add_subplot(121)

# xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')
# print('Data 2 taken')

# end = np.searchsorted(ts, end_t, side='left')
# for agent in range(k_elements):
#     plt.plot(ts[:end], xs[agent][:end], color=plot_colors[agent], label=f'agent {agent}')
# ax2.set_xlim(0, end_t)
# ax2.set_xlabel('t', fontsize=20)
# ax2.set_ylabel('x', fontsize=20)
# # ax2.set_xticks(fontsize=20)
# # ax2.set_yticks(fontsize=20)
# ax2.tick_params(axis='both', labelsize=20)
# ax2.legend(loc='lower right')

# path = '2024-09-09 14.52.28'
# xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')
# print('Data 1 taken')

# k_elements = 25
# ind = 17600

# plot_colors = mem.make_colors(k_elements)
# plt.figure(figsize=[6, 6])
# plt.subplots_adjust(left=0.2, right=0.99, top=0.96, bottom=0.135)
# for agent in range(len(xs)):
#     plt.plot(xs[agent][ind-20:ind+1], ys[agent][ind-20:ind+1], color=plot_colors[agent])
#     plt.scatter(xs[agent][ind], ys[agent][ind], color=plot_colors[agent])
# plt.xlabel("x", fontsize=28)
# # plt.ylabel("y", fontsize=28)
# plt.xticks([2, 4, 6], fontsize=24)
# plt.yticks([9, 7, 5], fontsize=24)
# plt.axis('equal')

# plt.show()

# make omega a graph
with open('./data/grid_experiments/2024-12-06 13.53.48 s_151_ tau_1/figsomegas.txt', 'r') as f:
    f_data = f.readlines()
data = f_data[0][1:-3].split(',')
data = [float(i) for i in data]
size = len(data)
a_arr = np.arange(0.15, 0.3001, 0.001)

plt.figure(figsize=[12, 4])
plt.plot(a_arr, data)
plt.xlabel('a', fontsize=28)
plt.ylabel(r'$\Omega$', fontsize=28)
plt.xticks([0.15, 0.2, 0.25, 0.3], fontsize=24)
plt.yticks([2.4, 2.0, 1.6, 1.2], fontsize=24)
plt.xlim(0.15, 0.30)

plt.show()