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

################################## make omega a graph
# with open('./data/grid_experiments/2024-12-06 13.53.48 s_151_ tau_1/figsomegas.txt', 'r') as f:
#     f_data = f.readlines()
# data = f_data[0][1:-3].split(',')
# data = [float(i) for i in data]
# size = len(data)
# a_arr = np.arange(0.15, 0.3001, 0.001)

# plt.figure(figsize=[12, 4])
# plt.plot(a_arr, data)
# plt.xlabel('a', fontsize=28)
# plt.ylabel(r'$\Omega$', fontsize=28)
# plt.xticks([0.15, 0.2, 0.25, 0.3], fontsize=24)
# plt.yticks([2.4, 2.0, 1.6, 1.2], fontsize=24)
# plt.xlim(0.15, 0.30)

# plt.show()

################################## график на плоскости x,y

########### parallel
# path = '2025-04-16 16.42.04 avg 1 T 0.3 coup__xy'
# time_moment_t = 183.00124
path = '2025-04-23 17.34.00 avg 1 T 0.3 10_coup_xy'
time_moment_t = 236.01912
tail_t = 1
fontsize = 20
colors=['green', 'blue']
xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')

time_moment_ind = np.searchsorted(ts, time_moment_t)
tail_ind = np.searchsorted(ts, time_moment_t - tail_t)
plt.plot(xs[0][tail_ind:time_moment_ind], ys[0][tail_ind:time_moment_ind], label='1', color=colors[0])
plt.plot(xs[1][tail_ind:time_moment_ind], ys[1][tail_ind:time_moment_ind], label='2', color=colors[1])
plt.scatter(xs[0][time_moment_ind], ys[0][time_moment_ind], color=colors[0])
plt.scatter(xs[1][time_moment_ind], ys[1][time_moment_ind], color=colors[1])
# plt.scatter(0, 0, color='black', marker='+')
plt.legend(loc='lower right')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.subplots_adjust(0.12, 0.12, 0.88, 0.88)
plt.axis('equal')
plt.show()

########## posl
path = '2025-04-23 15.45.37 avg 1 T 0.3 2_coup_xy'
time_moment_t = 320.0065
tail_t = 0.7
fontsize = 20
colors=['green', 'blue']
xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')

time_moment_ind = np.searchsorted(ts, time_moment_t)
tail_ind = np.searchsorted(ts, time_moment_t - tail_t)
plt.plot(xs[0][tail_ind:time_moment_ind+1], ys[0][tail_ind:time_moment_ind+1], label='1', color=colors[0])
plt.plot(xs[1][tail_ind:time_moment_ind+1], ys[1][tail_ind:time_moment_ind+1], label='2', color=colors[1])
plt.scatter(xs[0][time_moment_ind], ys[0][time_moment_ind], color=colors[0])
plt.scatter(xs[1][time_moment_ind], ys[1][time_moment_ind], color=colors[1])
# plt.scatter(0, 0, color='black', marker='+')
plt.legend(loc='upper right')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.subplots_adjust(0.12, 0.12, 0.88, 0.88)
# plt.axis('equal')
plt.show()


################################## последовательное и параллельное движение в зависимости от T
# T_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# R1 = 10
# R2 = 4
# phi_xy_arr_1 = [0.6847835068041368, -0.00024049863673546046, 0.016399686986767355, -0.0014154269900184453, -0.010997210275395797, 0.031097546132844593, 0.025533781682397793]
# phi_dxdy_arr_1 = [0.6375898236046866, 0.31648827953012715, -0.25693484402494227, 0.30962105818645375, 0.2743782522851758, -0.25893996038311884, -0.2457769594281952]
# A_arr_1 = [4.455478062375993, 1.5124195579256943, 1.7261598711883346, 2.2042859582012393, 2.453346620407209, 2.5493922605254595, 3.0109133651860933]
# phi_xy_arr_2 = [0.11376065961169111, 0.057988421391588676, 0.03933245767994036, 0.029402790892718744, 0.023948687061699826, 0.020171091165224735, 0.01726948444268236]
# phi_dxdy_arr_2 = [0.11271029305157491, 0.059480732721169244, 0.04093742135830397, 0.02980063878226318, 0.025376289001301476, 0.021079473748354595, 0.018059385011073077]
# A_arr_2 = [0.16318678883448445, 0.06658616698732485, 0.0383475592670401, 0.027268424041928974, 0.02005159101466687, 0.013773590437628437, 0.013545272988249132]

# plt.plot(T_arr[1:], phi_xy_arr_1[1:], label='параллельная связь 1')
# plt.plot(T_arr[1:], phi_dxdy_arr_1[1:], label='параллельная связь 2')
# plt.xlabel('t', fontsize=18)
# plt.ylabel(r'$\phi_1 - \phi_2$', fontsize=18)
# plt.legend()
# plt.grid()
# plt.show()

# plt.plot(T_arr[1:], phi_xy_arr_2[1:], label='последовательная связь 1')
# plt.plot(T_arr[1:], phi_dxdy_arr_2[1:], label='последовательная связь 2')
# plt.xlabel('t', fontsize=18)
# plt.ylabel(r'$\phi_1 - \phi_2$', fontsize=18)
# plt.legend()
# plt.grid()
# plt.show()

# plt.plot(T_arr[1:], A_arr_1[1:], label='параллельная связь')
# plt.xlabel('t', fontsize=18)
# plt.ylabel(r'$A_1 - A_2$', fontsize=18)
# plt.legend()
# plt.grid()
# plt.show()

# plt.plot(T_arr[1:], A_arr_2[1:], label='последовательная связь')
# plt.xlabel('t', fontsize=18)
# plt.ylabel(r'$A_1 - A_2$', fontsize=18)
# plt.legend()
# plt.grid()
# plt.show()