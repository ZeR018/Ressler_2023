import memory_worker as mem
import matplotlib.pyplot as plt
import numpy as np
import settings as s
import pandas as pd

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
# path = '2025-05-14 12.01.09 avg 1 T 0.3 10_coup_xy'
# time_moment_t = 247.52166
# path = '2025-05-16 12.34.12 avg 1 T 0.3 16_coup_y'
# time_moment_t = 167.00125
# grid c_dep_omega xy
# path = '2025-05-19 16.22.30 avg 1 T 0.3 16_coup_xy'
# time_moment_t = 45.6013
# time_moment_t = 70.4004
# grid c_dep_omega y
# path = '2025-05-20 10.09.54 avg 1 T 0.3 16_coup_y'
# time_moment_t = 65.30331
# parallel c y
# path = '2025-05-20 11.40.09 avg omega10_coup_y'
# time_moment_t = 378.00314
# path = '2025-05-20 12.15.44 avg omega10_coup_y'
# time_moment_t = 50.01777
# posl omega y
# path = '2025-05-20 12.22.20 avg omega10_coup_y'
# time_moment_t = 28.00242
# time_moment_t = 53.01605
# tail_t = 0.07
# fontsize = 20
# colors=['green', 'blue']
# xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')

# time_moment_ind = np.searchsorted(ts, time_moment_t)
# tail_ind = np.searchsorted(ts, time_moment_t - tail_t)
# plot_colors = mem.make_colors(len(xs))
# for agent in range(len(xs)):
#     plt.plot(xs[agent][tail_ind:time_moment_ind+1], ys[agent][tail_ind:time_moment_ind+1], color=plot_colors[agent], zorder=1)
# for agent in range(len(xs)):
#     plt.scatter(xs[agent][time_moment_ind], ys[agent][time_moment_ind], color=plot_colors[agent], zorder=2)
# # plt.scatter(0, 0, color='black', marker='+')
# # plt.legend()
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.subplots_adjust(0.18, 0.18, 0.94, 0.94)
# plt.xlabel('x', fontsize=fontsize)
# plt.ylabel('y', fontsize=fontsize)
# plt.xticks([-0.2, 0, 0.2, 0.4, -0.4], fontsize=fontsize)
# plt.yticks([3.38, 3.34, 3.3], fontsize=fontsize)
# # plt.axis('equal')
# plt.show()

################################################### vdp
# # path = '2025-05-27 13.59.28 vdp_17'
# # time_moment_t = 8.00906
# # time_moment_t = 114.50583
# path = '2025-05-27 14.51.14 vdp_17'
# # time_moment_t = 2.00698
# time_moment_t = 119.00244
# tail_t = 0.7
# fontsize = 20
# xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')
# time_moment_ind = np.searchsorted(ts, time_moment_t)
# tail_ind = np.searchsorted(ts, time_moment_t - tail_t)

# plot_colors = mem.make_colors(len(xs)-1)
# for agent in range(len(xs) - 1):
#     plt.plot(xs[agent][tail_ind:time_moment_ind+1], ys[agent][tail_ind:time_moment_ind+1], color=plot_colors[agent], zorder=1)
# for agent in range(len(xs) - 1):
#     plt.scatter(xs[agent][time_moment_ind], ys[agent][time_moment_ind], color=plot_colors[agent], zorder=2)
# # vdp
# plt.plot(xs[-1][tail_ind:time_moment_ind+1], ys[-1][tail_ind:time_moment_ind+1], color='red', label='VDP', zorder=1)
# plt.scatter(xs[-1][time_moment_ind], ys[-1][time_moment_ind], color='red', zorder=2)

# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.subplots_adjust(0.18, 0.18, 0.94, 0.94)
# plt.xlabel('x', fontsize=fontsize)
# plt.ylabel('y', fontsize=fontsize)
# plt.xticks([0, 0.2, 0.4, 0.6], fontsize=fontsize)
# plt.yticks([0.2, 0, -0.2, -0.4], fontsize=fontsize)
# plt.legend()
# # plt.axis('equal')
# plt.show()

########################################### Oscillatory death #######################################
path = '2025-06-02 15.27.39 avg omega10_coup_y'
t_end = 50
fontsize=20
xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')
t_end_ind = np.searchsorted(ts, t_end)
plot_colors = mem.make_colors(len(xs))
plt.figure(figsize=[8,4])
for agent in range(len(xs)):
    plt.plot(ts[:t_end_ind], ys[agent][:t_end_ind], color=plot_colors[agent])
plt.xlabel(r'$t$', fontsize=fontsize)
plt.ylabel(r'$y$', fontsize=fontsize)
plt.subplots_adjust(0.15, 0.22, 0.98, 0.96)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()

########### posl
# path = '2025-05-14 17.13.29 avg 1 T 0.3 2_coup_xy'
# time_moment_t = 180.00403
########## parall
# path = '2025-04-23 15.45.37 avg 1 T 0.3 2_coup_xy'
# time_moment_t = 320.0065
# tail_t = 0.7
# fontsize = 20
# colors=['green', 'blue']
# xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')

# time_moment_ind = np.searchsorted(ts, time_moment_t)
# tail_ind = np.searchsorted(ts, time_moment_t - tail_t)
# plt.plot(xs[0][tail_ind:time_moment_ind+1], ys[0][tail_ind:time_moment_ind+1], label='1', color=colors[0])
# plt.plot(xs[1][tail_ind:time_moment_ind+1], ys[1][tail_ind:time_moment_ind+1], label='2', color=colors[1])
# plt.scatter(xs[0][time_moment_ind], ys[0][time_moment_ind], color=colors[0])
# plt.scatter(xs[1][time_moment_ind], ys[1][time_moment_ind], color=colors[1])
# # plt.scatter(0, 0, color='black', marker='+')
# plt.legend(loc='upper right')
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.subplots_adjust(0.18, 0.18, 0.94, 0.94)
# plt.xlabel('x', fontsize=fontsize)
# # plt.ylabel('y', fontsize=fontsize)
# # plt.axis('equal')
# plt.show()

################################## Зависимость амплитуды и частоты от параметров a и c

fontsize = 20
default_path = f"{s.grid_experiments_path}/Неоднородность по параметру С"
path_name_a = "series_solo_omega_dep_a"
path_name_c = "series_solo_omega_dep_c"

# path = default_path + '/' + path_name_a
# data = pd.read_table(f"{path}/metrix.txt", sep=' ', header=None)
# # c_arr = data[0]
# a_arr = data[0]
# A_arr = data[1]
# omega_arr = data[2]
# size = len(a_arr)

# # plt.plot(c_arr, omega_arr)
# plt.plot(c_arr, A_arr)

# plt.xlim(c_arr[0], c_arr[size-1])
# plt.xlabel(r'$c$', fontsize=fontsize)

# # plt.ylabel(r'$\omega$', fontsize=fontsize)
# plt.ylabel(r'$A$', fontsize=fontsize)

# plt.xticks(fontsize=fontsize)
# plt.yticks([10, 9, 8, 7], fontsize=fontsize)
# plt.subplots_adjust(0.18, 0.18, 0.94, 0.94)
# plt.show()

# plt.plot(a_arr, omega_arr)
# # plt.plot(a_arr, A_arr)

# plt.xlim(a_arr[0], a_arr[size-1])
# plt.xlabel(r'$a$', fontsize=fontsize)

# plt.ylabel(r'$\omega$', fontsize=fontsize)
# # plt.ylabel(r'$A$', fontsize=fontsize)

# plt.xticks([0.16, 0.2, 0.24, 0.28], fontsize=fontsize)
# plt.yticks([2.4, 2.0, 1.6, 1.2], fontsize=fontsize)
# plt.subplots_adjust(0.18, 0.18, 0.94, 0.94)
# plt.show()

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

####################################################### График зависимости отношения средних частот к силе связи
# path = '2025-05-22 11.23.58 series_omega_dep_T_0.0_0.2'
# fontsize = 20

# # with open(f'{path}/omega_meta.txt')
# data = pd.read_table(f'{s.grid_experiments_path}{path}/omega_meta.txt', sep=' ', header=None)
# T_arr = data[0]
# size = len(T_arr)
# omega_ratios = data[1]

# plt.figure(figsize=[6, 3])
# plt.plot(T_arr, omega_ratios)
# plt.subplots_adjust(0.2, 0.25, 0.97, 0.94)
# plt.xlabel(r"$d'$", fontsize=fontsize)
# plt.ylabel(r"$\Omega_2/\Omega_1$", fontsize=fontsize)
# plt.xticks([0, 0.05, 0.1, 0.15, 0.2], fontsize=fontsize)
# plt.yticks([1, 1.01, 1.02, 1.03], fontsize=fontsize)
# plt.show()

###################################################### Зависимость средних частот от времени
# from couplings import calc_omegas_avg_mean
# # path = 'data/grid_experiments/Omegas/2025-05-21 15.46.41 avg omega10_coup_y'
# path = 'data/grid_experiments/2025-05-22 13.45.25 avg omega2_coup_y'
# _, _, _, ts = mem.read_integration_data(path + '/integration_data.txt')
# t_max = round(ts[-1], 3)
# k_elements = 2

# df = pd.read_table(f'./{path}/omegas_data.txt', sep=' ', header=None)

# omegas = [df.iloc[i].to_numpy() for i in range(k_elements)]

# plot_colors = mem.make_colors(k_elements)
# for i in range(k_elements):
#     omegas_mean, times_arr = calc_omegas_avg_mean(omegas[i], ts, t_max, time_skip=0)
#     plt.plot(times_arr, omegas_mean, color=plot_colors[i])
# plt.subplots_adjust(0.18, 0.18, 0.94, 0.94)
# plt.ylabel(r'$\omega$', fontsize=fontsize)
# plt.xlabel(r'$t$', fontsize=fontsize)
# plt.yticks([1.115, 1.11, 1.105, 1.1], fontsize=fontsize)
# plt.xticks([0, 10000, 20000, 30000, 40000], fontsize=fontsize)
# plt.show()
