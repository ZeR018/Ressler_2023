from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import time
from config import settings as s
from random import sample
from datetime import datetime
import main as m
import os
from matplotlib.animation import ArtistAnimation
from model import func_rossler_del_elems, generate_w_arr, function_rossler_and_VanDerPol, func_rossler_3_dim

t_max = s.t_max             # Время интегрирования

k_str = s.k_str             # Число агентов в одной строке
k_col = s.k_col             # Число агентов в одном столбце
k_elements = k_str * k_col  # Число агентов 
k = s.k                       # Число уравнений для одного агента (всегда 3)
radius = s.radius           # Радиус связи

small_animation = s.small_animation
full_animation = s.full_animation
need_save_last_state = s.need_save_last_state

stopping_borded_work = s.stopping_borded_work
stopping_border_radius  = s.stopping_border_radius

for_find_grid_IC = {'10x10': '2023-11-04 06.49.04', '10x10t': '37.41633',
                    '7x7': '2023-11-02 19.44.45', '7x7t': '81.99422',
                    '5x5': '2023-10-28 15.24.36', '5x5t': '122.92734',
                    '5x5_2': '2023-11-13 12.13.04', '5x5t_2': '80.90308',
                    '5x5_3': '2023-11-13 12.18.32', '5x5t_3': '90.9484',
                    '5x5_4': '2023-11-13 12.18.32', '5x5t_4': '59.69824'}


# Функции синхронизации

# Возвращает время с точностью до секунды
def hms_now(type = '0'):
    if type == '0':
        return str(datetime.now().time())[:-7]
    if type == 'ms':
        return [datetime.now().minute, datetime.now().second]
    if type == 'm':
        return datetime.now().minute


# Сохраняет начальные условия и массив частот(w) в указанный файл
def save_IC_and_w(IC, w, path, _k_elements = k_elements, _radius = radius, T = s.T):
    with open(path, 'w') as f:
        print(_k_elements, file=f)
        for i in range(_k_elements):
            print(IC[i*k], IC[i*k+1], IC[i*k+2], file=f)

        print(w, file=f)
        print('T:', T, file=f)
        print('k_col:', k_col, 'k_str:', k_str, file=f)
        print('r:', _radius, file=f)


# Считывает сохраненные НУ и w из указанного файла
def read_IC(path):
    IC = []
    with open(path, 'r') as f:
        f_data = f.readlines()

    size = int(f_data[0])

    for i in range(1, size+1):
        line = f_data[i].split()
        IC.append(float(line[0]))
        IC.append(float(line[1]))
        IC.append(float(line[2]))

    w_str = f_data[size + 1][1:-2].split()
    w = [float(i[:-1]) for i in w_str]
    w[-1] = float(w_str[-1])

    T_IC = float(f_data[size + 2][3:])

    k_str_col_split = f_data[size + 3].split(' ')
    k_col_ = int(k_str_col_split[1])
    k_str_ = int(k_str_col_split[3])

    if len(f_data) > size + 3 == True:
        radius_IC = float(f_data[size + 4][3:])
    else:
        radius_IC = s.radius

    return IC, w, T_IC, [k_col_, k_str_], radius_IC


# Сохраняет данные, полученные интегрированием в указанный файл
def save_integration_data(integration_data, path, _k_elements = k_elements):
    xs, ys, zs, ts = integration_data
    
    size = len(xs[0])
    with open(path, 'w') as f:
        for i in range(size):
            for agent in range(_k_elements):
                print(xs[agent][i], ys[agent][i], zs[agent][i], file=f)
        for i in range(size):
            print(ts[i], file=f)
        print(_k_elements, file=f)
        print(size, file=f)


# Считывает данные интегрирования из указанного файла
def read_integration_data(path):
       
    with open(path, 'r') as f:
        f_data = f.readlines()

    _k_elements = int(f_data[-2])
    size = int(f_data[-1])

    xs, ys, zs, ts = [], [], [], []
    for i in range(_k_elements):
        xs.append([])
        ys.append([])
        zs.append([])
    for i in range(size):
        for agent in range(_k_elements):
            line = f_data[i * _k_elements + agent].split()
            xs[agent].append(float(line[0]))
            ys[agent].append(float(line[1]))
            zs[agent].append(float(line[2]))
    for i in range(size*_k_elements, size*(_k_elements+1)):
        ts.append(float(f_data[i]))

    return [xs, ys, zs, ts]


# Сохраняет данные интегрирования, НУ, w, и все необходимые графики
def save_data(integration_data, IC, w, figs_arr = [], fig_names_arr = [], deleted_elems = [], k_elements = k_elements):
    date = str(datetime.now().date())
    time = hms_now().replace(':', '.')

    new_dir = s.grid_experiments_path + date + ' ' + time
    os.mkdir(new_dir)

    save_IC_and_w(IC, w,new_dir + '/IC.txt', _k_elements = k_elements)
    save_integration_data(integration_data, new_dir + '/integration_data.txt', _k_elements = k_elements)
    
    if figs_arr != []:
        for i in range(len(figs_arr)):
            figs_arr[i].savefig(new_dir + '/' + fig_names_arr[i] + '.png')

    # Для картинок
    data_dir = new_dir + '/data'
    os.mkdir(data_dir)

    if deleted_elems != []:
        save_grid_mask(deleted_elems, new_dir + '/deleted elems mask' + '.txt')

    return new_dir, data_dir

def make_frames_grid_agents(xs_arr, ys_arr, plot_colors, _k_elements = k_elements, frames_step = 60, deleted_elems = []):

    fig = plt.figure(figsize=[12,12])
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    fig.suptitle('Сетка мобильных агентов')

    # Кол-во кадров
    num_frames = len(xs_arr[0])
    frames = []

    for i in range(0, num_frames, frames_step):
        frame = []

        for agent in range(_k_elements):
            if deleted_elems.count(agent) > 0:
                continue
            
            line = 0
            if i < 500:
                line, = ax.plot(xs_arr[agent][:i], ys_arr[agent][:i], color=plot_colors[agent])
            else:
                line, = ax.plot(xs_arr[agent][i-500:i], ys_arr[agent][i-500:i], color=plot_colors[agent])
            frame.append(line)

            point = ax.scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
            frame.append(point)

            # Border
        if stopping_borded_work == True:
            n_points_circle = 10000
            x_circle = [ stopping_border_radius * np.cos(2 * np.pi * x / n_points_circle) for x in range(n_points_circle)]
            y_circle = [ stopping_border_radius * np.sin(2 * np.pi * x / n_points_circle) for x in range(n_points_circle)]

            circle, = ax.plot(x_circle, y_circle, color='gray')
            frame.append(circle)

        frames.append(frame)


    
    return frames, fig


def make_colors(_k_elements):
    res_col_arr = []
    color_index = 1

    color_step = 1 / _k_elements
    for i in range(_k_elements):
        res_col_arr.append((round(1 - color_index, 5), round(color_index, 5), round(1 - color_index, 5)))
        color_index -= color_step

    return res_col_arr

def draw_and_save_graphics_many_agents(xs_arr, ys_arr, ts_arr, path_save_graphs, plot_colors, _k_elements, 
                                       step_graphs=50, undeleted_elems = [], infotm_about_managing_agent = (), mashtab = [], grid=True):
    # infotm_about_managing_agent
    if infotm_about_managing_agent != ():
        xs_managing_agent_arr = xs_arr[-1]
        ys_managing_agent_arr = ys_arr[-1]
        managing_agent_name = infotm_about_managing_agent[0]
        managing_agent_color = infotm_about_managing_agent[1]

    
    if undeleted_elems == []:
        undeleted_elems = range(_k_elements)

    num_frames = len(xs_arr[0])

    for i in range(0+step_graphs, num_frames, step_graphs):
        plt.figure(figsize=[8,8])

        for agent in range(_k_elements):

            if(i < 101):
                plt.plot(xs_arr[agent][:i], ys_arr[agent][:i], color=plot_colors[agent])
                plt.scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
            else:
                plt.plot(xs_arr[agent][i-100:i], ys_arr[agent][i-100:i], color=plot_colors[agent])
                plt.scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])

        if mashtab != []:
            plt.xlim(mashtab[0], mashtab[1])
            plt.ylim(mashtab[2], mashtab[3])
        plt.xlabel('x')
        plt.ylabel('y')
        if grid: plt.grid()
        plt.suptitle(str(i) + ' time: ' + str(round(ts_arr[i], 5)))

        if infotm_about_managing_agent != ():
            if(i < 50):
                plt.plot(xs_managing_agent_arr[:i], ys_managing_agent_arr[:i], label=managing_agent_name, color=managing_agent_color)
            else:
                plt.plot(xs_managing_agent_arr[i-100:i], ys_managing_agent_arr[i-100:i], label=managing_agent_name, color=managing_agent_color)
            plt.scatter(xs_managing_agent_arr[i], ys_managing_agent_arr[i], color=managing_agent_color)
            plt.legend()

        if stopping_borded_work == True:
            n_points_circle = 10000
            x_circle = [ stopping_border_radius * np.cos(2 * np.pi * x / n_points_circle) for x in range(n_points_circle)]
            y_circle = [ stopping_border_radius * np.sin(2 * np.pi * x / n_points_circle) for x in range(n_points_circle)]

            plt.plot(x_circle, y_circle, color='gray')

        plt.savefig(path_save_graphs + '/' + '{:05}'.format(i) + '.png')
        plt.close()

    return 0

def binary_search_time(a, elem, occuracy):
    mid: int = len(a) // 2
    left: int = 0
    right: int = len(a) - 1

    while round(a[mid], occuracy) != elem and left <= right:
        if elem > round(a[mid], occuracy):
            left = mid + 1
        else:
            right = mid - 1
        mid = (left + right) // 2

    if left > right:
        print("Incorrect data for binary search")
    else:
        return mid

def find_grid_IC_from_integration_data(datapath, time):

    # Берем необходимую информацию о начальных условиях
    IC_data = read_IC(datapath + '/IC.txt')
    _ , w, T_IC, [k_col_IC, k_str_IC], _ = IC_data

    global T, k_col, k_str, k_elements
    T = T_IC
    k_col = k_col_IC
    k_str = k_str_IC

    # Ищем результат эксперимента в нужный момент времени
    integration_data = read_integration_data(datapath + '/integration_data.txt')
    # Binary search

    IC = []
    index_grid_IC = binary_search_time(integration_data[3], float(time), 5)
    for agent in range(k_elements):
        for j in range(k):
            IC.append(integration_data[j][agent][index_grid_IC])

    # plot_colors = make_colors(k_elements)
    # for agent in range(k_elements):
    #     plt.scatter(IC[agent*k], IC[agent*k + 1], color=plot_colors[agent])
    # plt.show()

    return IC, w

def show_grid_mask(deleted_elems, k_col = k_col, k_str = k_str):
    print('Deleted elems mask:')
    for i in range(k_col):
        print('\t', end='')
        for j in range(k_str):
            if deleted_elems.count(i*k_str + j) > 0:
                print(0, end=' ')
            else:
                print(1, end=' ')

        print('\t', end='\n')

def save_grid_mask(deleted_elems, path_save):
    if path_save != '0':
        with open(path_save, 'w') as f:
            print('Deleted elems mask:', file=f)
            for i in range(k_col):
                print('\t', end='', file=f)
                for j in range(k_str):
                    if deleted_elems.count(i*k_str + j) > 0:
                        print(0, end=' ', file=f)
                    else:
                        print(1, end=' ', file=f)
                print('\t', end='\n', file=f)


def pick_elements_for_delete(k_deleted_elements, k_col = k_col, k_str = k_str, type=1, pick_type = 'rand'):

    if pick_type == 'rand':
        return sample(range(k_col * k_str), k_deleted_elements)
    
    if k_col * k_str == 25:
        match(k_deleted_elements):
            case 1:
                return [12]
            case 2:
                return [11, 13]
            case 3:
                return [11, 12, 13]
            case 5:
                return [7, 11, 12, 13, 17]
            case 9:
                return [6, 7, 8, 11, 12, 13, 16, 17, 18]
    elif k_col * k_str == 36:
        match(k_deleted_elements):
            case 2:
                return [14, 15]
            case 4:
                return [14, 15, 20, 21]
            case 8:
                if type == 1:
                    return [8, 14, 15, 16, 19, 20, 21, 27]
                elif type == 2:
                    return [13, 14, 15, 16 , 19, 20, 21, 22]
                elif type == 3:
                    return [8, 9, 14, 15, 20, 21, 26, 27]
            case 12:
                return [8, 9, 13, 14, 15, 16, 19, 20, 21, 22, 26, 27]
            case 16:
                return [7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28]
            
    elif k_col * k_str == 49:
        match(k_deleted_elements):
            case 1:
                return [24]
            case 3:
                if type == 1:
                    return [23, 24, 25]
                elif type == 2:
                    return [17, 24, 31]
            case 5:
                if type == 1:
                    return [17, 23, 24, 25, 31]
                elif type == 2:
                    return [10, 17, 24, 31, 38]
                elif type == 3:
                    return [22, 23, 24, 25, 26]
            case 7:
                return [22, 23, 24, 25, 26, 17, 31]
            case 9:
                return [16, 17, 18, 23, 24, 25, 30, 31, 32]
            case 13:
                return [16, 17, 18, 23, 24, 25, 30, 31, 32, 10, 22, 26, 38]
            case 21:
                return [9, 10, 11, 16, 17, 18, 23, 24, 25, 30, 31, 32, 15, 22, 29, 19, 26, 33, 37, 38, 39]
            case 24:
                return [8, 12, 36, 40, 9, 10, 11, 16, 17, 18, 23, 24, 25, 30, 31, 32, 15, 22, 29, 19, 26, 33, 37, 38, 39]
            
    elif k_col * k_str == 100:
        match(k_deleted_elements):
            case 4:
                return [44, 45, 54, 55]
            case 8:
                return [44, 45, 54, 55, 34, 46, 53, 65]
            case 12:
                return [34, 35, 43, 44, 45, 46, 53, 54, 55, 56, 64, 65]
            case 16:
                return [33, 34, 35, 36, 43, 44, 45, 46, 53, 54, 55, 56, 63, 64, 65, 66]
            case 24:
                return [33, 34, 35, 36, 43, 44, 45, 46, 53, 54, 55, 56, 63, 64, 65, 66, 24, 25, 74, 75, 42, 52, 47, 57]
            case 32:
                return [23, 24, 25, 26, 33, 34, 35, 36, 43, 44, 45, 46, 53, 54, 55, 56, 63, 64, 65, 66, 73, 74, 75, 76, 32, 42, 52, 62, 37, 47, 57, 67]
            case 36:
                return [22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 77]
            case 44:
                return [14, 15, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 77, 84, 85]
            case 52:
                return [13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 
                        51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 83, 84, 85, 86]
            case 60:
                return [12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 
                        51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 82, 83, 84, 85, 86, 87]
            case 64:
                return [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 
                        51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88]
            

    print('functionality not implemented, k_elems = ' + str(k_elements) + ', k_deleted_elems = ' + str(k_deleted_elements))        

def plot_some_graph_without_grid(dir_path):
    [xs, ys, zs, ts] = read_integration_data(dir_path + '/integration_data.txt')

    new_dir = dir_path + '/graphs_without_grid'
    os.mkdir(new_dir)

    plot_colors = make_colors(len(xs))
    draw_and_save_graphics_many_agents(xs, ys, ts, new_dir,
                                       plot_colors, len(xs), grid=False)

    return 0

def mean_with_filter(array, min = 1000, max = 1000):
    sum = 0
    for item in array:
        if item > min and item < max:
            sum += item
    return sum



def make_experiment_delete_from_grid(k_deleted_elements, pick_type = 'rand', type = 1, type_IC = 'grid'):
    print('Start time:', hms_now())
    start_time = time.time()


    # Работаем только если элементов остается хотя бы на рамку для сетки
    if k_deleted_elements > k_col * k_str - (2 * k_col + 2 * k_str - 4):
        print('too mach elems to delete')
        return -1
    
    # Создаем "массив удаленных элементов"
    deleted_elems = pick_elements_for_delete(k_deleted_elements, type=type, pick_type=pick_type)
    if k_deleted_elements > 0:
        show_grid_mask(deleted_elems)

    # Берем НУ как состояние из другого эксперимента с сеткой
    IC, w = [], []
    if type_IC == 'rand':
        IC = m.generate_random_IC_ressler(5, 5, 0, k_elements)
        w = generate_w_arr(k_elements)
        T = s.T
    elif type_IC == 'grid':
        if k_col == 5 and k_str == 5:
            IC, w = find_grid_IC_from_integration_data("./data/grid_experiments/" + for_find_grid_IC['5x5'], for_find_grid_IC['5x5t'])
        elif k_col == 10 and k_str == 10:
            IC, w = find_grid_IC_from_integration_data("./data/grid_experiments/" + for_find_grid_IC['10x10'], for_find_grid_IC['10x10t'])

    # Задаем далекие НУ для убранных элементов - убираем элементы чтобы не мешались
    global undeleted_elems
    undeleted_elems = []
    for i in range(k_elements):
        if deleted_elems.count(i) > 0:
            if deleted_elems.count(i) > 1:
                print('wtf, deleted elems has broken: del_elems = ' + str(k_deleted_elements) + ', but count = ' + str(len(deleted_elems)))
                print('deleted elems:', deleted_elems)

            IC[i*k] = 10000
            IC[i*k + 1] = 10000
            IC[i*k + 2] = 10000
        else:
            undeleted_elems.append(i)

    start_solve_time = time.time()
    print('Start solve:', start_solve_time - start_time, 'time:', hms_now())

    sol = solve_ivp(func_rossler_del_elems, [0, t_max], IC, args=(k_elements, w, undeleted_elems, T), rtol=1e-11, atol=1e-11)
    print(f'res len: {len(sol.t)}')

    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t
    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', hms_now())

    plot_colors = make_colors(k_elements)
    
    gs_kw = dict(width_ratios=[1.5, 1], height_ratios=[1,1,1])
    fig, axd = plt.subplot_mosaic([['xt', 'yx',],
                                   ['yt', 'xz',],
                                   ['zt', 'yz',]],
                                gridspec_kw=gs_kw, figsize=(12, 8),
                                layout="constrained")
    for ax_n in axd:
        axd[ax_n].grid()
        axd[ax_n].set_xlabel(ax_n[1])
        axd[ax_n].set_ylabel(ax_n[0])
    fig.suptitle('Сетка мобильных агентов')

    for agent in range(k_elements):
        if deleted_elems.count(agent) > 0:
            continue

        axd['xt'].plot(ts, xs[agent], alpha=0.3, color=plot_colors[agent])
        axd['yt'].plot(ts, ys[agent], alpha=0.3, color=plot_colors[agent])
        axd['zt'].plot(ts, zs[agent], alpha=0.3, color=plot_colors[agent])
        
        axd['yx'].plot(xs[agent], ys[agent], alpha=0.3, color=plot_colors[agent])
        axd['xz'].plot(xs[agent], zs[agent], alpha=0.3, color=plot_colors[agent])
        axd['yz'].plot(zs[agent], ys[agent], alpha=0.3, color=plot_colors[agent])



    # plt.show()
    fig_last, ax_last = plt.subplots(figsize=[10, 6])
    for agent in range(k_elements):
        if deleted_elems.count(agent) > 0:
            continue
        ax_last.plot(xs[agent][-50:], ys[agent][-50:], color=plot_colors[agent])
        ax_last.scatter(xs[agent][-1], ys[agent][-1], color=plot_colors[agent])
    ax_last.grid()
    # plt.show()

    path_save, path_save_graphs = save_data([xs, ys, zs, ts], IC, w, [fig, fig_last], ['fig_graphs', 'fig_last_state'], deleted_elems=deleted_elems)

    draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 100, undeleted_elems)

    # Просто посмотреть первые 100 точек - как это работает
    # os.mkdir(path_save + '/first_100')
    # for i in range(100):
    #     plt.figure(figsize=[8,8])

    #     for agent in range(k_elements):
    #         if deleted_elems.count(agent) > 0:
    #             continue
    #         plt.plot(xs[agent][:i+1], ys[agent][:i+1], color=plot_colors[agent])
    #         plt.scatter(xs[agent][i], ys[agent][i], color=plot_colors[agent])

    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.grid()
    #     plt.suptitle(str(i) + ' time: ' + str(round(ts[i], 5)))

    #     plt.savefig(path_save + '/first_100' + '/t_' + str(i) + '.png')
    #     plt.close()

    # Анимация y(x)
    if small_animation:
        frames, fig_gif = make_frames_grid_agents(xs, ys, plot_colors, frames_step=20)
        interval = 40
        blit = True
        repeat = False
        animation = ArtistAnimation(
                    fig_gif,
                    frames,
                    interval=interval,
                    blit=blit,
                    repeat=repeat)
        animation_name = path_save + '/grid_agents_new'
        animation.save(animation_name + '.gif', writer='pillow')


    # # Анимация большая
    if full_animation:
        frames, frames_3d, fig, fig_3d = m.make_frames(xs, ys, zs, ts, 'Grid 5x5 agents', _k_elements = k_elements, frames_interval=50, plot_colors=plot_colors)
        # Задержка между кадрами в мс
        interval = 75
        # Использовать ли буферизацию для устранения мерцания
        blit = True
        # Будет ли анимация циклической
        repeat = False

        animation = ArtistAnimation(
                    fig,
                    frames,
                    interval=interval,
                    blit=blit,
                    repeat=repeat)

        animation_name = path_save + '/grid_agents_new_full'
        animation.save(animation_name + '.gif', writer='pillow')
        animation_3d = ArtistAnimation(
                    fig_3d,
                    frames_3d,
                    interval=interval,
                    blit=blit,
                    repeat=repeat)
        
        animation_3d.save(animation_name + '_3d.gif', writer='pillow')

    print('Other time', time.time() - time_after_integrate, 'time:', hms_now())

def rebuild_broken_system(stop_T, num_exps, broken_system_path):
    # Залезть в нужный эксперимент, достать оттуда начальные W, T и конечное состояние 
    # Выписать список удаленных элементов

    # Запустить num_exps экспериментов с разными Т в диапазоне [T_IC, stop_T]

    return 0

def make_experiment_use_vanderpol():
    print('Start time:', hms_now())

    global w
    w = generate_w_arr(k_elements, _range=[0.9, 1.1])

    start_time = time.time()

    # Берем НУ как состояние из другого эксперимента с сеткой
    IC, w = [], []
    if k_col == 5 and k_str == 5:
        IC, w = find_grid_IC_from_integration_data("./data/grid_experiments/" + for_find_grid_IC['5x5_4'], for_find_grid_IC['5x5t_4'])
    elif k_col == 10 and k_str == 10:
        IC, w = find_grid_IC_from_integration_data("./data/grid_experiments/" + for_find_grid_IC['10x10'], for_find_grid_IC['10x10t'])

    # Добавляем НУ Ван-дер-Поля в НУ
    IC.append(-1)
    IC.append(-1)
    IC.append(0)
    sol = solve_ivp(function_rossler_and_VanDerPol, [0, t_max], IC, args=(k_elements, w, -0.2, 1), rtol=1e-11, atol=1e-11)

    xs, ys, zs = [], [], []
    for i in range(k_elements + 1):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t

    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_time, 'time:', hms_now())

    plot_colors = make_colors(k_elements)

    path_save, path_save_graphs = save_data([xs, ys, zs, ts], IC, w)
    draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 50, 
                                       infotm_about_managing_agent=('Van der Pol', 'red'), mashtab=[-3, 3, -3, 3], grid=False)

    
    print('Other time', time.time() - time_after_integrate, 'time:', hms_now())

def make_experiment_with_controlling_agent(coeffs = [1, 1]):
    print('Start time:', hms_now())
    # Вид уравнения управляющего агента
    # dx/dt = a
    # dy/dt = b

    return 0



def make_grid_experiment():
    print('Start time:', hms_now())

    global w
    w = generate_w_arr(k_elements, _range=[0.9, 1.1])

    start_time = time.time()


    rand_IC = m.generate_random_IC_ressler(3., 3., 0.5, k_elements)
    sol = solve_ivp(func_rossler_3_dim, [0, t_max], rand_IC, rtol=1e-11, atol=1e-11)

    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t

    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_time, 'time:', hms_now())

    plot_colors = make_colors(k_elements)

    path_save, path_save_graphs = 0, 0
    if need_save_last_state:
        
        gs_kw = dict(width_ratios=[1.5, 1], height_ratios=[1,1,1])
        fig, axd = plt.subplot_mosaic([['xt', 'yx',],
                                    ['yt', 'xz',],
                                    ['zt', 'yz',]],
                                    gridspec_kw=gs_kw, figsize=(12, 8),
                                    layout="constrained")
        for ax_n in axd:
            axd[ax_n].grid()
            axd[ax_n].set_xlabel(ax_n[1])
            axd[ax_n].set_ylabel(ax_n[0])
        fig.suptitle('Сетка мобильных агентов')

        for agent in range(k_elements):
            axd['xt'].plot(ts, xs[agent], alpha=0.3, color=plot_colors[agent])
            axd['yt'].plot(ts, ys[agent], alpha=0.3, color=plot_colors[agent])
            axd['zt'].plot(ts, zs[agent], alpha=0.3, color=plot_colors[agent])
            
            axd['yx'].plot(xs[agent], ys[agent], alpha=0.3, color=plot_colors[agent])
            axd['xz'].plot(xs[agent], zs[agent], alpha=0.3, color=plot_colors[agent])
            axd['yz'].plot(zs[agent], ys[agent], alpha=0.3, color=plot_colors[agent])

        # plt.show()

        fig_last, ax_last = plt.subplots(figsize=[10, 6])
        for agent in range(k_elements):
            ax_last.plot(xs[agent][-50:], ys[agent][-50:], color=plot_colors[agent])
            ax_last.scatter(xs[agent][-1], ys[agent][-1], color=plot_colors[agent])
        ax_last.grid()
        # plt.show()

        path_save, path_save_graphs = save_data([xs, ys, zs, ts], rand_IC, w, [fig, fig_last], ['fig_graphs', 'fig_last_state'])

    else:
        path_save, path_save_graphs = save_data([xs, ys, zs, ts], rand_IC, w)

    # draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 100)

    # Анимация y(x)
    if small_animation:
        frames, fig_gif = make_frames_grid_agents(xs, ys, plot_colors, frames_step=20)
        interval = 40
        blit = True
        repeat = False
        animation = ArtistAnimation(
                    fig_gif,
                    frames,
                    interval=interval,
                    blit=blit,
                    repeat=repeat)
        animation_name = path_save + '/grid_agents_new'
        animation.save(animation_name + '.gif', writer='pillow')


    # # Анимация большая
    if full_animation:
        frames, frames_3d, fig, fig_3d = m.make_frames(xs, ys, zs, ts, 'Grid 4x5 agents', _k_elements = k_elements, frames_interval=100, plot_colors=plot_colors)
        # Задержка между кадрами в мс
        interval = 50
        # Использовать ли буферизацию для устранения мерцания
        blit = True
        # Будет ли анимация циклической
        repeat = False

        animation = ArtistAnimation(
                    fig,
                    frames,
                    interval=interval,
                    blit=blit,
                    repeat=repeat)

        animation_name = path_save + '/grid_agents_new_full'
        animation.save(animation_name + '.gif', writer='pillow')
        animation_3d = ArtistAnimation(
                    fig_3d,
                    frames_3d,
                    interval=interval,
                    blit=blit,
                    repeat=repeat)
        
        animation_3d.save(animation_name + '_3d.gif', writer='pillow')
    
    print('Other time', time.time() - time_after_integrate, 'time:', hms_now())

if __name__ == '__main__':

    make_experiment_delete_from_grid(0, type_IC='rand')
    make_experiment_delete_from_grid(0, type_IC='rand')
    make_experiment_delete_from_grid(0, type_IC='rand')
    make_experiment_delete_from_grid(0, type_IC='rand')

    make_experiment_delete_from_grid(0, type_IC='rand')
    make_experiment_delete_from_grid(0, type_IC='rand')
    make_experiment_delete_from_grid(0, type_IC='rand')
    make_experiment_delete_from_grid(0, type_IC='rand')
