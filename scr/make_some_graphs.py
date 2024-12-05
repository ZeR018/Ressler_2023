import memory_worker as mem
import matplotlib.pyplot as plt
import numpy as np
import settings as s


k_elements = 5
end_t = 3

fig = plt.figure(figsize=[12, 3])
plt.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.25)

path = 'lorenz_5_elements_sync'
xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')
print('Data 1 taken')

plot_colors = mem.make_colors(k_elements)
end = np.searchsorted(ts, end_t, side='left')
ax1 = fig.add_subplot(122)
for agent in range(k_elements):
    plt.plot(ts[:end], xs[agent][:end], color=plot_colors[agent], label=f'agent {agent}')
ax1.set_xlim(0, end_t)
ax1.set_xlabel('t', fontsize=20)
ax1.set_ylabel('x', fontsize=20)
# ax1.set_xticks(fontsize=20)
# ax1.set_yticks(fontsize=20)
ax1.tick_params(axis='both', labelsize=20)
ax1.legend(loc='lower right')
# ax1.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.25)


path = '2024-12-04 14.37.49 2lorens_r100_coup_y'
ax2 = fig.add_subplot(121)

xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')
print('Data 2 taken')

end = np.searchsorted(ts, end_t, side='left')
for agent in range(k_elements):
    plt.plot(ts[:end], xs[agent][:end], color=plot_colors[agent], label=f'agent {agent}')
ax2.set_xlim(0, end_t)
ax2.set_xlabel('t', fontsize=20)
# ax2.set_xticks(fontsize=20)
# ax2.set_yticks(fontsize=20)
ax2.tick_params(axis='both', labelsize=20)
ax2.legend(loc='lower right')

plt.show()