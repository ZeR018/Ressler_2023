from default_model import func_rossler_2_dim_params_maker, func_lorenz_params, Lorenz_params, Coup_params
from scipy.integrate import solve_ivp
import settings as s
from matplotlib import pyplot as plt
import memory_worker as mem
import numpy as np
import time
import joblib
from matplotlib.animation import ArtistAnimation

def one_exp_couplings(IC, w_arr, a, isSolo = True, couplings = (False, True, False), couplings_rep = (False, False, False),
                      k_elements = s.k_elements, t_max = s.t_max, tau = s.tau, T = s.T, small_animation = True, system = 'rossler', save_dir = s.grid_experiments_path, suffix=''):
    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())

    if system == 'lorenz':
        c = Coup_params(k_elements=k_elements, radius=60, T = 0.2, couplings=couplings)
        r = [166.1, 166.11, 166.12, 166.13, 166.14]
        # r = [28, 28.05, 28.1, 28.12, 28.15]
        l = Lorenz_params(r=r)
        func_rhs = func_lorenz_params(l, c)
        sol = solve_ivp(func_rhs, [0, t_max], IC, 
                    rtol=s.toch[0], atol=s.toch[1], method=s.method)
    else:
        func_rhs = func_rossler_2_dim_params_maker(k_elements, couplings, T_=T, couplings_rep=couplings_rep)
        sol = solve_ivp(func_rhs, [0, t_max], IC, args=(w_arr, a, tau), rtol=s.toch[0], atol=s.toch[1], method=s.method)
    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

    # Sort integration data 
    k = s.k
    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t

    plot_colors = mem.make_colors(k_elements)

    if isSolo:
        if system == 'lorenz':
            suffix += 'lorenz_'
        if T > 0:
            suffix += f"coup_{'x' if couplings[0] else ''}{'y' if couplings[1] else ''}{'z' if couplings[2] else ''}"
        elif T < 0:
            suffix += f"coup_{'-x' if couplings[0] else ''}{'-y' if couplings[1] else ''}{'-z' if couplings[2] else ''}"
        if couplings_rep[0] + couplings_rep[1] + couplings_rep[2] > 0:
            suffix += f"_{'x' if couplings_rep[0] else ''}{'y' if couplings_rep[1] else ''}{'z' if couplings_rep[2] else ''}"
        if small_animation:
            suffix += ' A'


        path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, w_arr, [], [], T=T, k_elements=k_elements, a=a, tau=tau, dir_name_suffix=suffix, path=save_dir)
        plt.close()
        lim_param = 20
        if system != 'lorenz':
            mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, t_step=0.5, 
                                               mashtab=[-lim_param, lim_param, -lim_param, lim_param] if couplings_rep[2] == 1 else [], num_prevs_elems=2)

        # Анимация y(x)
        if small_animation:
            # frames, fig_gif = mem.make_frames_grid_agents(xs, ys, ts, plot_colors, frames_step=20, _k_elements = k_elements, lims=[-lim_param, lim_param, -lim_param, lim_param])
            frames, fig_gif = mem.make_frames_n(xs, ys, ts, plot_colors, t_step=0.4, _k_elements = k_elements, lims=[-lim_param, lim_param, -lim_param, lim_param])
            interval = 60
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
            print(f'GIF made, time: {mem.hms_now()}')

        path_save_time_series = path_save + '/time_series'
        mem.make_dir(path_save_time_series)

        # Graphs xt
        size_xt_graph = 30
        num_graphs = int(t_max / size_xt_graph)
        for graph_ind in range(0, num_graphs):
            plt.figure(figsize=[16, 4])
            start = np.searchsorted(ts, size_xt_graph * graph_ind, side='left')
            end = np.searchsorted(ts, size_xt_graph * (graph_ind + 1), side="left")
            for agent in range(k_elements):
                plt.plot(ts[start:end], xs[agent][start:end], color=plot_colors[agent], label=f'agent {agent+1}')
            plt.grid()
            plt.ylim(-20, 20)
            plt.xlabel('t', fontsize=20)
            plt.ylabel('x', fontsize=20)
            plt.legend()
            plt.subplots_adjust(left=0.05, right=0.985, top=0.98, bottom=0.15)
            plt.savefig(path_save_time_series + f'/xt_{graph_ind}.png')
            plt.close()

        # Graph yt
        size_yt_graph = 30
        num_graphs = int(t_max / size_yt_graph)
        for graph_ind in range(0, num_graphs):
            plt.figure(figsize=[16, 4])
            start = np.searchsorted(ts, size_yt_graph * graph_ind, side='left')
            end = np.searchsorted(ts, size_yt_graph * (graph_ind + 1), side="left")
            for agent in range(k_elements):
                plt.plot(ts[start:end], ys[agent][start:end], color=plot_colors[agent], label=f'agent {agent+1}')
            plt.xlabel('t', fontsize=15)
            plt.ylabel('y', fontsize=15)
            plt.grid()
            plt.legend()
            plt.subplots_adjust(left=0.05, right=0.985, top=0.98, bottom=0.15)
            plt.savefig(path_save_time_series + f'/yt_{graph_ind}.png')
            plt.close()

        # Graph zt
        size_zt_graph = 30
        num_graphs = int(t_max / size_zt_graph)
        for graph_ind in range(0, num_graphs):
            plt.figure(figsize=[16, 4])
            start = np.searchsorted(ts, size_zt_graph * graph_ind, side='left')
            end = np.searchsorted(ts, size_zt_graph * (graph_ind + 1), side="left")
            for agent in range(k_elements):
                plt.plot(ts[start:end], zs[agent][start:end], color=plot_colors[agent], label=f'agent {agent+1}')
            plt.xlabel('t', fontsize=20)
            plt.ylabel('z', fontsize=20)
            plt.ylim(bottom=-2)
            plt.grid()
            plt.legend()
            plt.subplots_adjust(left=0.05, right=0.985, top=0.98, bottom=0.15)
            plt.savefig(path_save_time_series + f'/zt_{graph_ind}.png')
            plt.close()

    # Calc dists between agents
    dists = []
    for i in range(len(xs[0])):
        agents_mean_dists = []
        for agent in range(k_elements):
            one_agent_dists = []
            for agent_j in range(k_elements):
                sqrt = 0
                if agent != agent_j:
                    sqrt = np.sqrt((xs[agent][i] - xs[agent_j][i])**2 + (ys[agent][i] - ys[agent_j][i])**2)
                one_agent_dists.append(sqrt)
            agent_mean_dist = sum(one_agent_dists) / (k_elements - 1)
            agents_mean_dists.append(agent_mean_dist)
        dists.append(sum(agents_mean_dists) / k_elements)

    plt.figure(figsize=[16, 4])
    plt.plot(ts, dists)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('d')
    plt.savefig(path_save + '/dists.png')
    plt.close()

    last_state = []
    for agent in range(k_elements):
        last_state.append(xs[agent][-1])
        last_state.append(ys[agent][-1])
        last_state.append(zs[agent][-1])
    # return last_state, dists_summ # return last_state

    return last_state

def series_lorenz_dist_agents(IC, T_arr, couplings = (False, True, False), 
                      k_elements = s.k_elements, t_max = s.t_max, radius = 60):
    dir, xt_dir, dists_dir = mem.make_dirs_lorenz_exps(f"2lorens_r{radius}_coup_{'x' if couplings[0] else ''}{'y' if couplings[1] else ''}{'z' if couplings[2] else ''}")

    dists_by_t = []
    for T in T_arr:
        # Integrate
        start_solve_time = time.time()
        print(f'T = {T:.3f}', 'Start solve time:', mem.hms_now())

        c = Coup_params(k_elements=k_elements, radius=radius, T = T, couplings=couplings)
        # r = [166.1, 166.12, 166.14, 166.16, 166.18]
        r = [28, 28.05, 28.1, 28.12, 28.15]
        l = Lorenz_params(r=r)
        func_rhs = func_lorenz_params(l, c)
        sol = solve_ivp(func_rhs, [0, t_max], IC, 
                    rtol=s.toch[0], atol=s.toch[1], method=s.method)
    
        time_after_integrate = time.time()
        print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

        # Sort integration data 
        k = s.k
        xs, ys, zs = [], [], []
        for i in range(k_elements):
            xs.append(sol.y[i*k])
            ys.append(sol.y[i*k+1])
            zs.append(sol.y[i*k+2])
        ts = sol.t

        if len(T_arr) == 1:
            mem.save_integration_data([xs, ys, zs, ts], dir + '/integration_data.txt', _k_elements=k_elements)

        plot_colors = mem.make_colors(k_elements)

        def calc_dist_agents(x1, y1, z1, x2, y2, z2):
            dists = []
            for i in range(len(x1)):
                dists.append(np.sqrt((x1[i]-x2[i])**2 + (y1[i]-y2[i])**2 + (z1[i]-z2[i])**2))
            return dists
        dists = calc_dist_agents(xs[0], ys[0], zs[0], xs[1], ys[1], zs[1])
        start = np.searchsorted(ts, t_max - 100, side='left')
        dists_by_t.append(np.mean(dists[-100:])) # Для обобщения берем не просто последний результат, а считааем среднее расстояние в последних 10 точках

        # dists
        # Элементы для поиска
        # times_to_find_list = list(range(0, t_max, 10))
        # times_to_find = set(range(0, t_max, 10)) # Используем множество для быстрого поиска
        # # Получение индексов найденных элементов
        # indexes = list(filter(lambda index_element: index_element[1] in times_to_find, enumerate(ts))) # находить не точное число, а ближайшее к нему
        # # Извлечение только индексов
        # indexes_times = [index for index, element in indexes]
        # print(indexes_times)  # Вывод: индексы для массива ts, при которых ts[i]=10, 20, 30, ...

        # dists_look_at_times = [np.mean(dists[ts[times_to_find]])]
        # dists_look_at_times = []
        # for i, index in enumerate(indexes_times):
        #     start = index
        #     stop = indexes_times[i+1] if index != indexes_times[-1] else -1
        #     dists_look_at_times.append(np.mean(dists[start:stop]))

        step = 300
        ts_step = ts[step::step]
        dists_step = [np.mean(dists[i*step:(i+1)*step]) for i in range(len(ts) // step)]
        plt.figure(figsize=[12,4])
        # plt.plot(times_to_find_list.append(t_max), dists_look_at_times)
        plt.plot(ts_step, dists_step)
        plt.grid()
        plt.xlabel('t')
        plt.ylabel('Расст. между агентами')
        plt.title(f'Расстояние между агентами при T={T:.4f}')
        plt.savefig(dists_dir + f'/dist_{T:.4f}.png')
        plt.close()

        # Graphs xt
        # надо усреднять не по элементам, а по конкретным t
        size_xt_graph = 40
        num_graphs = int(t_max / size_xt_graph)
        for graph_ind in range(0, num_graphs):
            plt.figure(figsize=[24, 6])
            start = np.searchsorted(ts, size_xt_graph * graph_ind, side='left')
            end = np.searchsorted(ts, size_xt_graph * (graph_ind + 1), side="left")
            for agent in range(k_elements):
                plt.plot(ts[start:end], xs[agent][start:end], color=plot_colors[agent], label=f'agent {agent+1}')
            plt.grid()
            # plt.ylim(-20, 20)
            plt.xlabel('t')
            plt.ylabel('x')
            plt.legend()
            plt.savefig(xt_dir + f'/xt_{T:.4f}_{graph_ind}.png')
            plt.close()

    plt.figure(figsize=[12,4])
    plt.plot(T_arr, dists_by_t)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('Расст. между агентами')
    # plt.semilogx()
    plt.title('Зависимость расстояния между агентами от T')
    plt.savefig(dir + f'/dist_t.png')
    plt.close()

        
k_elements = 2
a = 0.22
# a_arr = [0.162, 0.202, 0.224]
# a = a_arr[0]

### Способ взятия начальных условий

## берем из файла с готовыми НУ
IC_fname = 'series_IC_20_20.txt'
IC_index = 3
IC_arr, w_arrs = mem.read_series_IC(s.temporary_path + IC_fname)
w_arr = w_arrs[IC_index:IC_index+k_elements]
IC = IC_arr[IC_index][:k_elements*s.k]


## берем случайные
# IC = mem.generate_random_IC_ressler(5, 5, 1, k_elements)
# w_arr = []

# ## Запуск одного эксперимента с настройкой связей
# last_state = one_exp_couplings(IC, w_arr, a, couplings=(0, 1, 0), k_elements=k_elements, t_max=205, tau=1, T=s.T, small_animation=False, system = 'rossler')

# ## берем результат предыдущего эксперимента (синхронизированные агенты) и запускаем с другими связями
# T = 0.3
# one_exp_couplings(last_state, w_arr, a, couplings=(0, 1, 1), k_elements=k_elements, t_max=210, tau=1, T=T, small_animation=False, system = 'rossler')


## Просто один эксперимент с какой-то связью
T = 0.3
s.toch = [1e-10, 1e-10]
date = mem.hms_now().replace(':', '-')
path = mem.make_dir(s.grid_experiments_path + f'{date} 5 el couplings')
path += '/'
t_max = 1000

print('Старт программы', f'Результаты в {path}')
# # Связь 1
# one_exp_couplings(IC, w_arr, a, couplings=(1, 0, 0), couplings_rep=(0, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# one_exp_couplings(IC, w_arr, a, couplings=(0, 1, 0), couplings_rep=(0, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 1), couplings_rep=(0, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# one_exp_couplings(IC, w_arr, a, couplings=(1, 1, 0), couplings_rep=(0, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)

# # Связь 2
# print('2')
# one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 0), couplings_rep=(1, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 0), couplings_rep=(0, 1, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 0), couplings_rep=(0, 0, 1), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# s.toch = [1e-07, 1e-07]
# one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 0), couplings_rep=(1, 1, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# s.toch = [1e-12, 1e-12]

# # Связь 3 
# print('3')
# T = -0.3
# one_exp_couplings(IC, w_arr, a, couplings=(1, 0, 0), couplings_rep=(0, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# one_exp_couplings(IC, w_arr, a, couplings=(0, 1, 0), couplings_rep=(0, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 1), couplings_rep=(0, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)
# T = -0.3
# s.toch = [1e-09, 1e-09]
# one_exp_couplings(IC, w_arr, a, couplings=(1, 1, 0), couplings_rep=(0, 0, 0), k_elements=k_elements, 
#                                t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)

# далее
print('last')
k_elements = 3
IC = IC_arr[IC_index][:k_elements*s.k]
w_arr = w_arrs[IC_index:IC_index+k_elements]
T = 0.3
s.toch = [1e-07, 1e-07]
one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 0), couplings_rep=(1, 1, 0), k_elements=k_elements, 
                               t_max=t_max, tau=1, T=T, small_animation=True, system = 'rossler', save_dir=path)

## Серия экспериментов со связью при разных T + график какой-то
# T_arr = np.arange(20, 100, 10)
# dists_arr = []
# for T in T_arr:
#     print(f'---------------- T = {T}')
#     last_state, dist = one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 1), k_elements=k_elements, t_max=500, tau=1, T=T, small_animation=False, system = 'rossler')
#     dists_arr.append(dist)

# plt.plot(T_arr, dists_arr)
# plt.grid()
# plt.show()

### Серия экспериментов с изменением параметра силы связи

# T_arr = np.logspace(-2, 1, num=31)
# T_arr = np.arange(4., 6., 0.2)
# T_arr = [4]
# series_lorenz_dist_agents(IC, T_arr, k_elements=k_elements, t_max=500, couplings=(0, 1, 0), radius=100)
# T = 5