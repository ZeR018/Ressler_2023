from datetime import datetime
import os
import settings as s
from matplotlib import pyplot as plt
import numpy as np
from random import uniform

stopping_borded_work = s.stopping_borded_work
stopping_border_radius  = s.stopping_border_radius
k_str = s.k_str             # Число агентов в одной строке
k_col = s.k_col             # Число агентов в одном столбце
k_elements = k_str * k_col  # Число агентов 

# Проверяет существование файла
def exists(path):
    try:
        file = open(path)
    except IOError as e:
        return False
    else:
        return True

# Создаем массив частот(w) для всех агентов
def generate_w_arr(k_elements, _range=[0.93, 1.07]):
    w_arr = []
    for i in range(k_elements):
        w_arr.append(round(uniform(_range[0], _range[1]), 3))
    return w_arr

# Возвращает время с точностью до секунды
def hms_now(type = '0'):
    if type == '0':
        return str(datetime.now().time())[:-7]
    if type == 'ms':
        return [datetime.now().minute, datetime.now().second]
    if type == 'm':
        return datetime.now().minute

# Сохраняет начальные условия и массив частот(w) в указанный файл
def save_IC_and_w(IC, w, path, _k_elements = k_elements, _radius = s.radius, T = s.T, a = s.a, tau = s.tau, k = s.k):
    with open(path, 'w') as f:
        print(_k_elements, file=f)
        for i in range(_k_elements):
            for param in range(k):
                print(IC[i*k + param], file=f, end='\t')
            print('', file=f)

        print(w, file=f)
        print('a: ', a, file=f)
        print('T:', T, file=f)
        print('k_col:', k_col, 'k_str:', k_str, file=f)
        print('r:', _radius, file=f)
        print('tau:', tau, file=f)


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

    if len(f_data) > size + 4 == True:
        tau = float(f_data[size + 5][5:])

    return IC, w, T_IC, [k_col_, k_str_], radius_IC, tau


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
def save_data(integration_data, IC, w, figs_arr = [], fig_names_arr = [], deleted_elems = [], k_elements = k_elements, 
              a = s.a, T = s.T, tau = s.tau, dir_name_suffix = '', k = s.k, save_int_data = True):
    date = str(datetime.now().date())
    time = hms_now().replace(':', '.')

    suffix = ' ' + dir_name_suffix if dir_name_suffix != '' else ''
    new_dir = s.grid_experiments_path + date + ' ' + time + suffix
    os.mkdir(new_dir)

    save_IC_and_w(IC, w,new_dir + '/IC.txt', _k_elements = k_elements, a=a, T = T, tau = s.tau, k = k)

    if save_int_data:
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

def make_dir_for_series_experiments(w, a, n_exps, IC_file_name, dop_names = {}, mod = ''):
    date_without_year = str(datetime.now().date())
    time = hms_now().replace(':', '.')

    if mod != '':
        new_dir = f'{s.grid_experiments_path + date_without_year} {time} s_{n_exps}'
    else:
        new_dir = f'{s.grid_experiments_path + date_without_year} {time} s_{n_exps}_{mod}'

    for k, v in dop_names.items():
        new_dir += f' {k}_{v}'
    os.mkdir(new_dir)

    # Save main IC data
    with open(new_dir + '/IC.txt', 'w') as f:
        print('w: ', w, file=f)
        print('a: ', a, file=f)
        print('IC file name: ', IC_file_name, file=f)
        print('Num experiments: ', n_exps, file=f)
        print('r: ', s.radius, file=f)
        print('T:', s.T, file=f)

    figs_dir = new_dir + '/figs'
    os.mkdir(figs_dir)

    times_dir = new_dir +'/times.txt'
    with open(times_dir, 'w') as f:
            print('',  file=f, end='')

    return new_dir, figs_dir, times_dir

def make_frames_grid_agents(xs_arr, ys_arr, plot_colors, _k_elements = k_elements, frames_step = 60, deleted_elems = [], lims = []):

    fig = plt.figure(figsize=[12,12])
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    fig.suptitle('Сетка мобильных агентов')

    if lims != []:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])

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

    print(f'Frames made, time: {hms_now()}')
    
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
                                       step_graphs=50, undeleted_elems = [], inform_about_managing_agent = (), mashtab = [], grid=True):
    # infotm_about_managing_agent
    if inform_about_managing_agent != ():
        xs_managing_agent_arr = xs_arr[-1]
        ys_managing_agent_arr = ys_arr[-1]
        managing_agent_name = inform_about_managing_agent[0]
        managing_agent_color = inform_about_managing_agent[1]

    
    if undeleted_elems == []:
        undeleted_elems = range(_k_elements)

    num_frames = len(xs_arr[0])

    for i in range(0+step_graphs, num_frames, step_graphs):
        plt.figure(figsize=[8,8])

        for agent in range(_k_elements):

            if(i < 101):
                plt.plot(xs_arr[agent][:i+1], ys_arr[agent][:i+1], color=plot_colors[agent])
                plt.scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
            else:
                plt.plot(xs_arr[agent][i-100:i+1], ys_arr[agent][i-100:i+1], color=plot_colors[agent])
                plt.scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])

        if mashtab != []:
            plt.xlim(mashtab[0], mashtab[1])
            plt.ylim(mashtab[2], mashtab[3])
        plt.xlabel('x')
        plt.ylabel('y')
        if grid: plt.grid()
        plt.suptitle(str(i) + ' time: ' + str(round(ts_arr[i], 5)))

        if inform_about_managing_agent != ():
            if(i < 50):
                plt.plot(xs_managing_agent_arr[:i], ys_managing_agent_arr[:i], label=managing_agent_name, color=managing_agent_color)
            else:
                plt.plot(xs_managing_agent_arr[i-100:i+1], ys_managing_agent_arr[i-100:i+1], label=managing_agent_name, color=managing_agent_color)
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
    _ , w, T_IC, [k_col_IC, k_str_IC], _, _ = IC_data

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
        for j in range(s.k):
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


def plot_some_graph_without_grid(dir_path):
    [xs, ys, zs, ts] = read_integration_data(dir_path + '/integration_data.txt')

    new_dir = dir_path + '/graphs_without_grid'
    os.mkdir(new_dir)

    plot_colors = make_colors(len(xs))
    draw_and_save_graphics_many_agents(xs, ys, ts, new_dir,
                                       plot_colors, len(xs), grid=False)

    return 0


def make_frames(xs_arr, ys_arr, zs_arr, ts_arr, graph_title, graph_3d_title = '', _k_elements = k_elements, frames_interval = 60, plot_colors=s.plot_colors):
    if graph_3d_title == '':
        graph_3d_title = graph_title

    # Пытаемся расположить графики
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
    fig.suptitle(graph_title)
    
    fig_3d = plt.figure(figsize=[8,8])
    ax_3d = fig_3d.add_subplot(projection='3d')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
    ax_3d.grid()
    fig_3d.suptitle(graph_3d_title)

    # Кол-во кадров
    num_frames = len(xs_arr[0])
    
    frames = []
    frames_3d = []
    # Создаем каждый кадр для gif
    for i in range(0, num_frames, frames_interval):
        frame = []
        frame_3d = []

        # Цикл по каждому агенту (элементу)
        for agent in range(_k_elements):

            
            # Очень старая линия почти не видима, чтобы не мешалась (только на графиках xt, xyz)
            if i > 2000:
                # xy
                xy_line_very_transperent, = axd['yx'].plot(xs_arr[agent][:i-2000], ys_arr[agent][:i-2000], color=plot_colors[agent], alpha=0.15)
                xy_line_transperent, = axd['yx'].plot(xs_arr[agent][i-2001:i-500], ys_arr[agent][i-2001:i-500], color=plot_colors[agent], alpha=0.4)
                xy_line, = axd['yx'].plot(xs_arr[agent][i-501:i], ys_arr[agent][i-501:i], color=plot_colors[agent])
                frame.append(xy_line_very_transperent)
                frame.append(xy_line_transperent)
                frame.append(xy_line)

                # xz
                xz_line_very_transperent, = axd['xz'].plot(zs_arr[agent][:i-2000], xs_arr[agent][:i-2000], color=plot_colors[agent], alpha=0.15)
                xz_line_transperent, = axd['xz'].plot(zs_arr[agent][i-2001:i-500], xs_arr[agent][i-2001:i-500], color=plot_colors[agent], alpha=0.4)
                xz_line, = axd['xz'].plot(zs_arr[agent][i-501:i], xs_arr[agent][i-501:i], color=plot_colors[agent])
                frame.append(xz_line_very_transperent)
                frame.append(xz_line_transperent)
                frame.append(xz_line)

                # yz
                yz_line_very_transperent, = axd['yz'].plot(zs_arr[agent][:i-2000], ys_arr[agent][:i-2000], color=plot_colors[agent], alpha=0.15)
                yz_line_transperent, = axd['yz'].plot(zs_arr[agent][i-2001:i-500], ys_arr[agent][i-2001:i-500], color=plot_colors[agent], alpha=0.4)
                yz_line, = axd['yz'].plot(zs_arr[agent][i-501:i], ys_arr[agent][i-501:i], color=plot_colors[agent])
                frame.append(yz_line_very_transperent)
                frame.append(yz_line_transperent)
                frame.append(yz_line)

                xyz_line_very_transperent, = ax_3d.plot3D(xs_arr[agent][:i-2000], ys_arr[agent][:i-2000], zs_arr[agent][:i-2000], color=plot_colors[agent], alpha=0.15)
                xyz_line_transperent, = ax_3d.plot3D(xs_arr[agent][i-2001:i-500], ys_arr[agent][i-2001:i-500], zs_arr[agent][i-2001:i-500], color=plot_colors[agent], alpha=0.4)
                xyz_line, = ax_3d.plot3D(xs_arr[agent][i-501:i], ys_arr[agent][i-501:i], zs_arr[agent][i-501:i], color=plot_colors[agent])
                frame_3d.append(xyz_line_very_transperent)
                frame_3d.append(xyz_line_transperent)
                frame_3d.append(xyz_line)

            elif i > 500:
                #xy
                xy_line_transperent, = axd['yx'].plot(xs_arr[agent][:i-500], ys_arr[agent][:i-500], color=plot_colors[agent], alpha=0.4)
                xy_line, = axd['yx'].plot(xs_arr[agent][i-501:i], ys_arr[agent][i-501:i], color=plot_colors[agent])
                frame.append(xy_line_transperent)
                frame.append(xy_line)
                
                #xz
                xz_line_transperent, = axd['xz'].plot(zs_arr[agent][:i-500], xs_arr[agent][:i-500], color=plot_colors[agent], alpha=0.4)
                xz_line, = axd['xz'].plot(zs_arr[agent][i-501:i], xs_arr[agent][i-501:i], color=plot_colors[agent])
                frame.append(xz_line_transperent)
                frame.append(xz_line)
                
                #yz
                yz_line_transperent, = axd['yz'].plot(zs_arr[agent][:i-500], ys_arr[agent][:i-500], color=plot_colors[agent], alpha=0.4)
                yz_line, = axd['yz'].plot(zs_arr[agent][i-501:i], ys_arr[agent][i-501:i], color=plot_colors[agent])
                frame.append(yz_line_transperent)
                frame.append(yz_line)

                #xyz
                xyz_line_transperent, = ax_3d.plot3D(xs_arr[agent][:i-500], ys_arr[agent][:i-500], zs_arr[agent][:i-500], color=plot_colors[agent], alpha=0.4)
                xyz_line, = ax_3d.plot3D(xs_arr[agent][i-501:i], ys_arr[agent][i-501:i], zs_arr[agent][i-501:i], color=plot_colors[agent])
                frame_3d.append(xyz_line_transperent)
                frame_3d.append(xyz_line)

            else: 
                #xy
                xy_line, = axd['yx'].plot(xs_arr[agent][:i], ys_arr[agent][:i], color=plot_colors[agent])
                frame.append(xy_line)
                #xz
                xy_line, = axd['xz'].plot(zs_arr[agent][:i], xs_arr[agent][:i], color=plot_colors[agent])
                frame.append(xy_line)
                #yz
                xy_line, = axd['yz'].plot(zs_arr[agent][:i], ys_arr[agent][:i], color=plot_colors[agent])
                frame.append(xy_line)
                #xyz
                xyz_line, = ax_3d.plot3D(xs_arr[agent][:i], ys_arr[agent][:i], zs_arr[agent][:i], color=plot_colors[agent])
                frame_3d.append(xyz_line)

            # Последние точки xy, xz, yz
            xy_point = axd['yx'].scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
            xz_point = axd['xz'].scatter(zs_arr[agent][i], xs_arr[agent][i], color=plot_colors[agent])
            yz_point = axd['yz'].scatter(zs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
            frame.append(xy_point)
            frame.append(xz_point)
            frame.append(yz_point)
            #xyz
            xyz_point = ax_3d.scatter(xs_arr[agent][i], ys_arr[agent][i], zs_arr[agent][i], color=plot_colors[agent])
            frame_3d.append(xyz_point)

            # Рисуем графики x(t), y(t), z(t)
            xt_line, = axd['xt'].plot(ts_arr[:i], xs_arr[agent][:i], color=plot_colors[agent])
            yt_line, = axd['yt'].plot(ts_arr[:i], ys_arr[agent][:i], color=plot_colors[agent])
            zt_line, = axd['zt'].plot(ts_arr[:i], zs_arr[agent][:i], color=plot_colors[agent])
            frame.append(xt_line)
            frame.append(yt_line)
            frame.append(zt_line)

            # Рисуем последние точки x(t), y(t), z(t)
            xt_point = axd['xt'].scatter(ts_arr[i], xs_arr[agent][i], color=plot_colors[agent])
            yt_point = axd['yt'].scatter(ts_arr[i], ys_arr[agent][i], color=plot_colors[agent])
            zt_point = axd['zt'].scatter(ts_arr[i], zs_arr[agent][i], color=plot_colors[agent])
            frame.append(xt_point)
            frame.append(yt_point)
            frame.append(zt_point)
            
        frames.append(frame)
        frames_3d.append(frame_3d)

    return frames, frames_3d, fig, fig_3d

def generate_random_IC_ressler(x_max, y_max, z_max, k_elems=k_elements, x_min='',  y_min='',  z_min=''):
    if x_min == '':
        x_min = - x_max
    if y_min == '':
        y_min = - y_max
    if z_min == '':
        z_min = - z_max
    
    res_IC_arr = []
    for i in range(k_elems):
        res_IC_arr.append(uniform(x_min, x_max))
        res_IC_arr.append(uniform(y_min, y_max))
        res_IC_arr.append(uniform(z_min, z_max))

    return res_IC_arr

def generate_and_write_series_IC(generator_params: tuple, n_exps = 100, 
                                 k_elements = k_elements, path = s.temporary_path, file_name = ' ', w_arr = []):
    gp = generator_params

    if w_arr == []:
        w_arr = generate_w_arr(k_elements)

    if file_name == ' ':
        file_name = f'series_IC_{n_exps}_{k_elements}.txt'
    full_path = path + file_name

    # Смотрим, есть ли путь с таким названием
    is_path_new = False
    idx = 1
    while not is_path_new:
        if exists(full_path):
            if idx == 1:
                full_path = full_path[:-4] + f'({idx})' + '.txt'
            else:
                num_symbols_idx = int(np.log10(idx))
                full_path = full_path[:-num_symbols_idx-4] + f'({idx})' + '.txt'

            idx += 1
        else:
            is_path_new = True


    with open(full_path, 'w') as f:
        # Generate and save IC for any exp
        for i in range(n_exps):
            IC = generate_random_IC_ressler(gp[0], gp[1], gp[2], k_elements)
            for c in IC:
                print(f'{c} ', file=f, end='')
            print('', file=f)
        
        # Save w_arr
        for w in w_arr:
            print(f'{w} ', file=f, end='')
        print('', file=f)

        # Save params
        print(k_elements, file=f)
        print(n_exps, file=f)
    return full_path

def read_series_IC(path):
    k = s.k
    with open(path, 'r') as f:
        f_data = f.readlines()

    w_str = f_data[-3].split(' ')[:-1]
    w_arr = [float(i) for i in w_str]

    _k_elements = int(f_data[-2])
    n_exps = int(f_data[-1])
        
    IC_arr = [[] for i in range(n_exps)]
    for i in range(n_exps):
        line = f_data[i].split()
        for j in range(_k_elements * k):
            IC_arr[i].append(float(line[j]))
    return IC_arr, w_arr

def read_times_series_experiments(path, new_value_for_nsl = 550, look_at_nsl = True):
    with open(path, 'r') as f:
        f_data = f.readlines()
    
    res = []
    for d in f_data:
        line = d.split(' ')             # Line делится на номер и остальное
        index = int(line[0])
        
        line = line[1].split('\t')      # остальная часть линии делится на value и nls(not synchronizated last time) при наличии
        value = float(line[0])
        
        if not len(line) == 1:          # если у нас есть nsl, надо его обработать (пока не обрабатываем)
            nsl = line[1][:-1]
            if look_at_nsl: 
                value = new_value_for_nsl
        res.append(value)
    return res