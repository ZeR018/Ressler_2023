import numpy as np
from matplotlib import pyplot as plt
import settings as s
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

# Изменить график
# a_dir_name = '100 а028'
# tau_dir_name = '01'
# full_dir = f'./data/grid_experiments/series_a_tau/{a_dir_name}/{tau_dir_name}'
# times_of_sync = mem.read_times_series_experiments(full_dir + '/times.txt')

# plt.figure()
# plt.hist(times_of_sync, np.arange(-10, 190, 10))
# plt.grid()
# # plt.xlim(-20, 0)
# plt.xlabel('Время синхронизации')
# plt.ylabel('Число синхронизаций')
# plt.savefig(full_dir + '/times_hist2.png')

# plt.show()


########################################################## Чтение и изменение всех гистограмм в папке
import os
def change_all_hists_in_dir():
    # рассматриваемая директория
    directory = './data/grid_experiments/series_a_tau/simple'

    dirs_a = []             # подпапки основной директории
    a_dirs_values = []      # в названии папок значения параметра а для эксперимента. массив для записи параметров а
    n_exp_dirs_values = []  # в названии папок значения параметра n_exps для эксперимента. массив для записи параметров n_exps

    # записывает все подпапки в директории
    for dir_a in os.scandir(directory):
        if dir_a.is_dir():
            dirs_a.append(dir_a.path)

    tau_dirs_values = []    # второй уровень подпапок имеет значение tau. массив для сохраннения его значений
    times_paths = []        # массив для всех найденных путей times

    for dir_a in dirs_a:
        for dir_tau in os.scandir(dir_a):
            # если на втором уровне директории только файлы (нет подпапок) (500 экспериментов)
            if dir_tau.is_file():
                if dir_tau.name == 'times.txt':
                    tau_dirs_values.append(1)
                    times_paths.append(dir_tau.path)
                    a_dirs_values.append('a022')
                    n_exp_dirs_values.append(dir_a.split('\\')[1])

            # если на втором уровне директории папки "tau"
            if dir_tau.is_dir():
                dir_a_name = dir_a.split('\\')[1]
                for dir_last in os.scandir(dir_tau.path):
                    if dir_last.is_file():
                        if dir_last.name == 'times.txt':
                            tau_dirs_values.append(dir_tau.name)
                            times_paths.append(dir_last.path)
                            a_dirs_values.append(dir_a_name.split(' ')[1])
                            n_exp_dirs_values.append(dir_a.split('\\')[1].split(' ')[0])
                            
    new_hists_dir = directory + '/hists'
    try:
        os.mkdir(new_hists_dir)
    except Exception as e:
        print('Hists dir is already exists')

    for i in range(len(times_paths)):
        # print(times_paths[i], tau_dirs_values[i], a_dirs_values[i], n_exp_dirs_values[i])
        times_of_sync = mem.read_times_series_experiments(times_paths[i], look_at_nsl=False)
        for ti in range(len(times_of_sync)):
            if times_of_sync[ti] == -10:
                times_of_sync[ti] = 520

        
        # probability theory
        mean = np.mean(times_of_sync)
        sigma = np.var(times_of_sync)
        median = np.median(times_of_sync)
        path = times_paths[i].removesuffix('times.txt') + 'analysis.txt'
        with open(path, 'w') as f:
            print(f'mean: {mean}', file=f)
            print(f'dispersion: {sigma}', file=f)
            print(f'median: {median}', file=f)

        # plot
        plt.figure()
        h = np.append(np.arange(0, 520, 20), 550)
        colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
        n, bins, patches = plt.hist(times_of_sync, h, edgecolor='darkblue')
        plt.xlim(-10, 530)
        plt.xlabel('Время синхронизации')
        plt.ylabel('Число синхронизаций')
        plt.title(f'mean = {round(mean, 2)}, d = {round(sigma, 2)}, median = {round(median, 2)}')
        
        plt.savefig(f'{new_hists_dir}/hist_{a_dirs_values[i]}_{tau_dirs_values[i]}_{n_exp_dirs_values[i]}.png')
        plt.close()




def change_time_220_in_file_and_compress():
    path = './data/grid_experiments/series_a_tau/simple/1000 a028/05'
    times_of_sync = mem.read_times_series_experiments(path + '/times.txt', look_at_nsl=False)
    for ti in range(len(times_of_sync)):
        if times_of_sync[ti] == -10:
            times_of_sync[ti] = 530

    with open(path + '/times_compressed.txt', 'w') as f:
        for ti in range(len(times_of_sync)):
            print(times_of_sync[ti], file=f)


# change_all_hists_in_dir()
change_time_220_in_file_and_compress()


# shapes = [
#     # tau = 0.5, 1, 2, 5, 10
#     [1.587, 1.283, 1.1133, 0.953, 0.897982550], # a = 0.16
#     [1.837, 1.8275, 1.84, 1.837, 1.851], # a = 0.22
#     [1.829, 1.7985, 1.795, 1.788, 1.78]  # a = 0.28
# ]

# rates = [
#     # tau = 1, 2, 5, 10
#     [0.0066, 0.009, 0.01448, 0.028, 0.050152811], # a = 0.16
#     [0.0148, 0.029, 0.06, 0.1474, 0.299], # a = 0.22
#     [0.01126, 0.022, 0.044, 0.109, 0.216]  # a = 0.28
# ]

# taus = [0.5, 1., 2., 5., 10]
# a_arr = [0.16, 0.22, 0.28]

# for i, a_i in enumerate(a_arr):
#     plt.plot(shapes[i], rates[i], label=f'a = {a_i}')
# plt.grid()
# plt.xlabel('shapes')
# plt.ylabel('rates')
# plt.legend()
# plt.show()


# from scipy.stats import gamma
# path = './data/grid_experiments/series_a_tau/1000/1000 a022/10'
# times_of_sync = mem.read_times_series_experiments(path + '/times.txt')
# for ti in range(len(times_of_sync)):
#     if times_of_sync[ti] == -10:
#         times_of_sync[ti] = 220
# plt.figure()
# h = np.append(np.arange(0, 210, 10), 250)
# colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
# n, bins, patches = plt.hist(times_of_sync, h, edgecolor='darkblue')
# plt.xlim(-10, 230)
# plt.xlabel('Время синхронизации')
# plt.ylabel('Число синхронизаций')
# plt.savefig(path + '/times_hist.png')
# plt.show()


# График плотности распределения с параметрами из R (метод макс. правдоподобия)
# shape, scale = 2.53477792, 0.03262352  # mean=4, std=2*sqrt(2)
# s = np.random.gamma(shape, scale, 1000)
# import matplotlib.pyplot as plt
# import scipy.special as sps  
# count, bins, ignored = plt.hist(s, 50, density=True)
# y = bins**(shape-1)*(np.exp(-bins/scale) /  
#                      (sps.gamma(shape)*scale**shape))
# print(y)
# plt.plot(bins, y, linewidth=2, color='r')  
# plt.show()
