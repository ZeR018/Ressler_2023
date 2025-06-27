from default_model import func_rossler_2_dim_params_maker, func_lorenz_params, function_rossler_2dim_vdp, Lorenz_params, Coup_params, Rossler_params
from scipy.integrate import solve_ivp
import settings as s
from matplotlib import pyplot as plt
import memory_worker as mem
import numpy as np
import time
import joblib
from matplotlib.animation import ArtistAnimation
from typing import Union
from itertools import product
import pandas as pd
from datetime import datetime

def calc_dist_between_agents(xs, ys, zs, ts):
    k_elements = len(xs)
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
    return dists

t_step_xy_graphs = 0.5
size_time_graph = 30

def calc_phi_omega_amplitude_for_agent(xs, ys, zs, ts, w, a, c, b = s.b):
    size = len(ts)
    phi_yx = []
    phi_dydx = []
    omega = []
    omega_yx = []
    A = []

    for t in range(size):
        x = xs[t]
        y = ys[t]
        z = zs[t]
        phi_i1 = np.arctan2(( w * x + a * y), (- w * y - z))
        phi_i2 = np.arctan2(y, x)
        phi_yx.append(phi_i2)
        phi_dydx.append(phi_i1)
        
        omega_i_dyddx = - (w*x + a*y) * (-w**2*x-w*a*y-b-z*(x-c))
        omega_i_ddydx = (-w*y-z) * (-w**2*y-w*z+a*w*x+a**2*y)

        omega_i_zn = (w*y+z)**2 + (w*x + a*y)**2
        omega.append((omega_i_ddydx + omega_i_dyddx)/omega_i_zn)

        dx = - w * y - z
        dy = w * x + a * y
        omega_yx_i = (dy * x - y * dx) / (x**2 + y**2)
        omega_yx.append(omega_yx_i)

        A.append(np.sqrt(x**2+y**2))

    return phi_yx, phi_dydx, omega, omega_yx, A

def calc_omegas_avg_mean(omega, ts, t_max, time_skip, first_step_t = 400, step_t = 400):
    omega_avg_arr = []
    times_arr = []
    t_curr = first_step_t + time_skip
    while(t_curr <= t_max):
        i_curr = np.searchsorted(ts, t_curr)
        t_prev = t_curr - first_step_t
        i_prev = np.searchsorted(ts, t_prev)
        omega_i = np.mean(omega[:i_curr])
        omega_avg_arr.append(omega_i)
        times_arr.append(t_curr)
        
        t_curr += step_t

    return omega_avg_arr, times_arr

def plot_omegas(omegas, ts, k_elements, path_save, save_name = "graph_omegas_mean"):
    plt.close()
    plt.figure()
    plot_colors = mem.make_colors(k_elements)
    for agent in range(k_elements):
        plt.plot(ts, omegas[agent], color=plot_colors[agent])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\omega$")
    plt.grid()
    plt.savefig(f"{path_save}/{save_name}.png")
    plt.show()
    plt.close()

def phase_converter(phase):
    if phase > 0:
        if phase > np.pi:
            phase = phase - 2 * np.pi
            phase_converter(phase)
        return phase
    else:
        if phase < - np.pi:
            phase = phase + 2 * np.pi
            phase_converter(phase)
        return phase
    
def rossler_lorenz_solver(system, sys_params, t_max, IC, couplings, couplings_rep):
    # Для системы Лоренца пока нерабочая версия - нужно изменить код для sys_params
    k_elements = sys_params.k_elements
        
    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now(), '\tmethod: ', s.method)

    if system == 'lorenz':
        c = Coup_params(k_elements=k_elements, radius=60, T = 0.2, couplings=couplings)
        r = [166.1, 166.11, 166.12, 166.13, 166.14]
        # r = [28, 28.05, 28.1, 28.12, 28.15]
        l = Lorenz_params(r=r)
        func_rhs = func_lorenz_params(l, c)
        sol = solve_ivp(func_rhs, [0, t_max], IC, 
                    rtol=s.toch[0], atol=s.toch[1], method=s.method)
    elif system == 'rossler_vdp':
        func_rhs = function_rossler_2dim_vdp(couplings, couplings_rep=couplings_rep, params=sys_params)
        sol = solve_ivp(func_rhs, [0, t_max], IC, rtol=s.toch[0], atol=s.toch[1], method=s.method)
    else:
        func_rhs = func_rossler_2_dim_params_maker(couplings, couplings_rep=couplings_rep, params=sys_params)
        sol = solve_ivp(func_rhs, [0, t_max], IC, rtol=s.toch[0], atol=s.toch[1], method=s.method)
    
    time_after_integrate = time.time()
    print(f'Integrate time: {(time.time() - start_solve_time):0.1f}s', 'time:', mem.hms_now(), '\tsize: ', len(sol.t))

    # Sort integration data 
    k = s.k
    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t
    return xs, ys, zs, ts, time_after_integrate

def one_exp_couplings(IC, isSolo = True, couplings = (False, True, False), couplings_rep = (False, False, False), 
                      sys_params : Union[Rossler_params, Lorenz_params] = Rossler_params() , 
                      t_max = s.t_max, small_animation = True, system = 'rossler', save_dir = s.grid_experiments_path, 
                      suffix='', step_t_for_avg = 1, time_skip = 0, integrate_data_path = None):
    
    print('w', sys_params.w_arr, 'c', sys_params.c)
    
    k_elements = sys_params.k_elements
    if not integrate_data_path:
        xs, ys, zs, ts, time_after_integrate = rossler_lorenz_solver(system, sys_params, t_max, IC, couplings, couplings_rep)
    else: 
        xs, ys, zs, ts = mem.read_integration_data(integrate_data_path)
        time_after_integrate = time.time()
        
    plot_colors = mem.make_colors(k_elements)

    if isSolo:
        suffix += f'{k_elements}_'
        if system == 'lorenz':
            suffix += 'lorenz_'
        if sys_params.T > 0:
            suffix += f"coup_{'x' if couplings[0] else ''}{'y' if couplings[1] else ''}{'z' if couplings[2] else ''}"
        elif sys_params.T < 0:
            suffix += f"coup_{'-x' if couplings[0] else ''}{'-y' if couplings[1] else ''}{'-z' if couplings[2] else ''}"
        if couplings_rep[0] + couplings_rep[1] + couplings_rep[2] > 0:
            suffix += f"_{'x' if couplings_rep[0] else ''}{'y' if couplings_rep[1] else ''}{'z' if couplings_rep[2] else ''}"
        if small_animation:
            suffix += ' A'

        path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, sys_params.w_arr, [], [], T=sys_params.T, k_elements=k_elements, a=sys_params.a, tau=sys_params.tau, c=sys_params.c, dir_name_suffix=suffix, path=save_dir)
        plt.close()
        lim_param = 20
        if system != 'lorenz':
            if not integrate_data_path:
                mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, t_step=t_step_xy_graphs, 
                                               mashtab=[-lim_param, lim_param, -lim_param, lim_param] if couplings_rep[2] == 1 else [], num_prevs_elems=2)
            else:
                with open(f"{path_save}/data.txt", 'w') as f:
                    print("Original figs on path:", file=f)
                    print(integrate_data_path, file=f)

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

        if not integrate_data_path:
            # Graphs xt
            size_xt_graph = size_time_graph
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
            size_yt_graph = size_time_graph
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
            size_zt_graph = size_time_graph
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

    time_after_save_data = time.time()
    print(f'Start analysis {(time.time() - time_after_integrate):0.1f}s', 'time:', mem.hms_now())

    # Calc dists between agents
    dists = calc_dist_between_agents(xs, ys, zs, ts)

    plt.figure(figsize=[16, 4])
    plt.plot(ts, dists)
    plt.grid()
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.11)
    plt.xlabel('t')
    plt.ylabel('d')
    plt.savefig(path_save + '/dists.png')
    plt.close()

    # Выбрасываем переходный процесс
    index_time_skip = np.searchsorted(ts, time_skip)
    ts_short = ts[index_time_skip:]
    xs_short = [x[index_time_skip:] for x in xs]
    ys_short = [y[index_time_skip:] for y in ys]
    zs_short = [z[index_time_skip:] for z in zs]

    # Calc phase and omega
    phases_yx = []
    phases_dydx = []
    omegas_short = []
    omegas_yx_short = []
    As_short = []
    omegas_avg_short = []
    omegas_yx_avg_short = []
    times_for_omegas_avg_short= []
    for agent in range(k_elements):
        phases_agent_yx, phases_agent_dydx, omegas_agent, omegas_yx_agent, As_agent = calc_phi_omega_amplitude_for_agent(xs_short[agent], ys_short[agent], zs_short[agent], ts_short, sys_params.w_arr[agent], sys_params.a, c=sys_params.c[agent])
        phases_yx.append(phases_agent_yx)
        phases_dydx.append(phases_agent_dydx)
        omegas_short.append(omegas_agent)
        omegas_yx_short.append(omegas_yx_agent)
        As_short.append(As_agent)

        omega_mean_agent, t_omega_mean_agent = calc_omegas_avg_mean(omegas_agent, ts, t_max, time_skip)
        omegas_avg_short.append(omega_mean_agent)
        times_for_omegas_avg_short = t_omega_mean_agent

        omegas_mean_yx_agent, t_omega_mean_yx_agent = calc_omegas_avg_mean(omegas_yx_agent, ts, t_max, time_skip)
        omegas_yx_avg_short.append(omegas_mean_yx_agent)
    plot_omegas(omegas_avg_short, times_for_omegas_avg_short, k_elements, path_save)
    plot_omegas(omegas_short, ts_short, k_elements, path_save, "omegas")
    plot_omegas(omegas_yx_avg_short, times_for_omegas_avg_short, k_elements, path_save, 'omegas_mean_yx' )
    plot_omegas(omegas_yx_short, ts_short, k_elements, path_save, "omegas_yx")
    
    omegas_short_df = pd.DataFrame(omegas_short)
    omegas_short_df.to_csv(f"{path_save}/omegas_data.txt", sep=' ', header=False, index=False)
    with open(f"{path_save}/omegas_metrix.txt", 'w') as f:
        print('avg', file=f)
        for agent in range(k_elements):
            print(agent, np.mean(omegas_short[agent]), file=f)
        print('last', file=f)
        for agent in range(k_elements):
            print(agent, omegas_short[agent][-1], file=f)


    if k_elements == 2:
        for step_t_for_avg in range(1, 2):

            # усредняем фазы, омеги и амплитуды и находим разность
            num_t = int((t_max - time_skip) / step_t_for_avg)
            phases_avg_yx = [[] for i in range(k_elements)]
            phases_avg_dydx = [[] for i in range(k_elements)]
            omegas_avg = [[] for i in range(k_elements)]
            As_avg = [[] for i in range(k_elements)]
            zs_avg = [[] for i in range(k_elements)]
            ts_avg = []
            i_prev = 0
            
            phases_diff_yx = []
            phases_diff_dydx = []
            omegas_diff = []
            As_diff = []
            zs_diff = []
            for ind in range(num_t):
                t_i = time_skip + step_t_for_avg * (ind + 1)
                i = np.searchsorted(ts_short, t_i)

                # усредняем
                ts_avg.append(t_i)
                for agent in range(k_elements):
                    phases_avg_yx[agent].append(np.mean(phases_yx[agent][i_prev:i]))
                    phases_avg_dydx[agent].append(np.mean(phases_dydx[agent][i_prev:i]))
                    omegas_avg[agent].append(np.mean(omegas_short[agent][i_prev:i]))
                    As_avg[agent].append(np.mean(As_short[agent][i_prev:i]))
                    zs_avg[agent].append(np.mean(zs_short[agent][i_prev:i]))
                
                # находим разности
                phases_diff_yx.append(phase_converter(phases_avg_yx[0][ind] - phases_avg_yx[1][ind]))
                phases_diff_dydx.append(phase_converter(phases_avg_dydx[0][ind] - phases_avg_dydx[1][ind]))
                omegas_diff.append(omegas_avg[0][ind] - omegas_avg[1][ind])
                As_diff.append(abs(As_avg[0][ind] - As_avg[1][ind]))
                zs_diff.append(zs_avg[0][ind] - zs_avg[1][ind])

                i_prev = i

            def plot_timeline_graph(x : list, t : list, ylabel : str, save_name : str, 
                                figsize_ : list=[12, 3], xlabel : list='t', 
                                subplots_adjust : list=[0.11, 0.97, 0.225, 0.97], path_save=path_save, 
                                format : str='.png', x2 : list=[], font_size : int = 18, graph_type : str = 'plot', grid=True) -> None:
                plt.figure(figsize=figsize_)
                plt.subplots_adjust(left=subplots_adjust[0], right=subplots_adjust[1], bottom=subplots_adjust[2], top=subplots_adjust[3])
                if graph_type == 'step':
                    if x2 == []:
                        plt.step(t, x, where='post')
                    else:
                        plt.step(t, x, label='1', where='post')
                        plt.step(t, x2, label='2', where='post')
                        plt.legend()
                else:
                    if x2 == []:
                        plt.plot(t, x)
                    else:
                        plt.plot(t, x, label='1')
                        plt.plot(t, x2, label='2')
                        plt.legend()
                if grid:
                    plt.grid()
                plt.xlabel(xlabel, fontsize=font_size)
                plt.ylabel(ylabel, fontsize=font_size)
                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                plt.savefig(path_save + '/' + save_name + '.png')
                plt.close()

            phases_diff_without_avg_yx = []
            phases_diff_without_avg_dydx = []
            omegas_diff_without_avg = []
            As_diff_without_avg = []
            for i in range(len(ts_short)):
                As_diff_without_avg.append(abs(As_short[0][i] - As_short[1][i]))
                phases_diff_without_avg_yx.append(phase_converter(phases_yx[0][i] - phases_yx[1][i]))
                phases_diff_without_avg_dydx.append(phase_converter(phases_dydx[0][i] - phases_dydx[1][i]))
                omegas_diff_without_avg.append(omegas_short[0][i] - omegas_short[1][i])

            print("Phases diff average arctg(y/x): ", np.mean(phases_diff_without_avg_yx))
            print("Phases diff average arctg(y'/x'): ", np.mean(phases_diff_without_avg_dydx))
            print('Amplitudes diff average: ', np.mean(As_diff))

            plot_timeline_graph(np.abs(phases_diff_yx), ts_avg, r'$|<\phi_2> - <\phi_1>|$', f'phases_avg_diff_yx_{step_t_for_avg}', graph_type='step', grid=False)
            plot_timeline_graph(np.abs(phases_diff_dydx), ts_avg, r'$|<\phi_2> - <\phi_1>|$', f'phases_avg_diff_dydx_{step_t_for_avg}', graph_type='step', grid=False)
            plot_timeline_graph(phases_diff_without_avg_yx, ts_short, r'$|\phi_2 - \phi_1|$', f'phases_diff_yx_{step_t_for_avg}', grid=False)
            plot_timeline_graph(phases_diff_without_avg_dydx, ts_short, r'$\phi_2 - \phi_1$', f'phases_diff_dydx_{step_t_for_avg}', grid=False)

            plot_timeline_graph(zs_diff, ts_avg, r'$<z_2> - <z_1>$', f'zs_diff_avg_{step_t_for_avg}', graph_type='step', grid=False)
            plot_timeline_graph(omegas_diff, ts_avg, r'$<\omega_2> - <\omega_1>$', f'omegas_avg_diff_{step_t_for_avg}', graph_type='step', grid=False)
            plot_timeline_graph(omegas_diff_without_avg, ts_short, r'$\omega_2 - \omega_1$', f'omegas_diff_{step_t_for_avg}', grid=False)
            plot_timeline_graph(As_diff, ts_avg, r'|$<A_2> - <A_1>$|', f'As_avg_diff_{step_t_for_avg}', graph_type='step', grid=False)
            plot_timeline_graph(As_short[0], ts_short, r'$A_2, A_1$', 'As', x2=As_short[1], grid=False)
            plot_timeline_graph(As_diff_without_avg, ts_short, r'$|A_2-A_1|$', 'As_diff', grid=False)

    # last state
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

def exp_omega_dep_T(IC, T_arr, couplings = (False, True, False), couplings_rep = (False, False, False), 
                      sys_params: Rossler_params = Rossler_params(), t_max = s.t_max, save_dir=s.grid_experiments_path):
    k_elements = sys_params.k_elements

    date = str(datetime.now().date())
    time = mem.hms_now().replace(':', '.')
    path_save = f"{save_dir}{date} {time} series_omega_dep_T_{T_arr[0]}_{T_arr[-1]}"
    mem.make_dir(path_save)

    omega_experiments = []
    omegas_ratios = []
    path_save_omegas_data = mem.make_dir(f"{path_save}/omegas")
    path_save_omegas_meta = f"{path_save}/omega_meta.txt"
    path_save_xyzt = mem.make_dir(f"{path_save}/time_series")
    for T in T_arr:
        T = round(T, 3)
        print('T', T)
        sys_params.T = T
        xs, ys, zs, ts, time_after_integrate = rossler_lorenz_solver("rossler", sys_params, t_max, IC, couplings, couplings_rep)

        omegas = []
        A = []
        for agent in range(k_elements):
            phi_yx_agent, phi_dydx_agent, omega_agent, omega_yx_agent, A_agent = calc_phi_omega_amplitude_for_agent(xs[agent], 
                                                            ys[agent], zs[agent], ts, sys_params.w_arr[agent], sys_params.a, sys_params.c[agent])
            omegas.append(omega_agent)
            A.append(A_agent)
        omega_experiments.append(omegas)

        if k_elements == 2:
            mean_omega_0 = np.mean(omegas[0])
            mean_omega_1 = np.mean(omegas[1])
            if mean_omega_0 < mean_omega_1:
                omegas_ratio = mean_omega_1/mean_omega_0
            else:
                omegas_ratio = mean_omega_0/mean_omega_1
            omegas_ratios.append(omegas_ratio)

        omegas_df = pd.DataFrame(omegas)
        omegas_df.to_csv(f"{path_save_omegas_data}/omegas_{T}.txt",sep=' ', header=False, index=False)
        with open(path_save_omegas_meta, 'a') as f:
            print(T, omegas_ratio, file=f)

        t_last_50 = np.searchsorted(ts, t_max-50)
        plt.plot(ts[t_last_50:], xs[0][t_last_50:])
        plt.plot(ts[t_last_50:], xs[1][t_last_50:])
        plt.xlabel('t')
        plt.ylabel('x')
        plt.grid()
        plt.savefig(f"{path_save_xyzt}/xt_{T}_last50.png")
        plt.close()

        plt.plot(ts[t_last_50:], ys[0][t_last_50:])
        plt.plot(ts[t_last_50:], ys[1][t_last_50:])
        plt.xlabel('t')
        plt.ylabel('y')
        plt.grid()
        plt.savefig(f"{path_save_xyzt}/yt_{T}_last50.png")
        plt.close()

        plt.plot(ts[t_last_50:], zs[0][t_last_50:])
        plt.plot(ts[t_last_50:], zs[1][t_last_50:])
        plt.xlabel('t')
        plt.ylabel('z')
        plt.grid()
        plt.savefig(f"{path_save_xyzt}/zt_{T}_last50.png")
        plt.close()

    plt.figure()
    plt.plot(T_arr, omegas_ratios)
    plt.xlabel(r"$d'$")
    plt.ylabel(r"$\Omega_2/\Omega_1$")
    plt.grid()
    plt.savefig(f"{path_save}/omegas_ratio_dep_d.png")
    plt.show()
    plt.close()

def one_exp_couplings_wdp(IC, couplings = (False, True, False), sys_params : Union[Rossler_params, Lorenz_params] = Rossler_params(), 
                      k_elements = s.k_elements, t_max = s.t_max):
    xs, ys, zs, ts, time_after_integrate = rossler_lorenz_solver("rossler_vdp", sys_params, t_max, IC, couplings, (0, 0, 0))
    path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, sys_params.w_arr, [], [], T=sys_params.T, k_elements=k_elements, 
                                                a=sys_params.a, tau=sys_params.tau, c=sys_params.c, dir_name_suffix=f"vdp_{k_elements}", path=s.grid_experiments_path)
    plot_colors = mem.make_colors(16)
    plot_colors.append('red')
    mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, sys_params.k_elements, t_step=t_step_xy_graphs, 
                                               num_prevs_elems=2)


def experiment_vdp():
    path = '2025-05-19 16.22.30 avg 1 T 0.3 16_coup_xy'
    xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')

    p = Rossler_params()
    p.k_elements = 17
    IC = []
    time_moment_t = 45.6013
    time_moment_ind = np.searchsorted(ts, time_moment_t)
    for i in range(p.k_elements-1):
        IC.append(xs[i][time_moment_ind])
        IC.append(ys[i][time_moment_ind])
        IC.append(zs[i][time_moment_ind])

    # vdp
    IC.append(0.0)
    IC.append(1.4)
    IC.append(0)

    C_arr_default = [8.50, 8.501, 8.502, 8.503]
    w_arr_default = [0.9, 0.95, 1, 1.05]
    decart = list(product(C_arr_default, w_arr_default))
    c_arr = []
    w_arr = []
    for elem in decart:
        c_arr.append(elem[0])
        w_arr.append(elem[1])
    p.c = c_arr
    p.w_arr = w_arr

    p.a = 0.16
    p.radius = 10
    t_max = 400
    one_exp_couplings_wdp(IC, (0, 1, 0), sys_params=p, k_elements=p.k_elements, t_max=t_max)

def calc_amplityde_for_vdp(path = '2025-05-27 14.51.14 vdp_17'):
    full_path = s.grid_experiments_path + path
    xs, ys, zs, ts = mem.read_integration_data(s.grid_experiments_path + path + '/integration_data.txt')
    k_elements = 17
    plot_colors = mem.make_colors(16)
    As = []
    fontsize = 20
    t_300 = np.searchsorted(ts, 300)
    for agent in range(16):
        # Сразу векторно
        As.append(np.sqrt(np.square(xs[agent][:t_300]) + np.square(ys[agent][:t_300])))
    
        plt.plot(ts[:t_300], As[agent], color = plot_colors[agent])
    plt.xlabel(r'$t$', fontsize=fontsize)
    plt.ylabel(r'$A$', fontsize=fontsize)
    plt.xticks([0, 100, 200, 300], fontsize=fontsize)
    plt.yticks([4, 3 ,2, 1, 0], fontsize=fontsize)
    plt.subplots_adjust(0.18, 0.18, 0.94, 0.94)

    plt.show()


def experiments1():
    p = Rossler_params()
    p.k_elements = 10
    t_max = 4000
    p.a = 0.16
    p.T = 0.3
    s.toch = [1e-11, 1e-11]
    # a_arr = [0.162, 0.202, 0.224]
    # a = a_arr[0]

    ### Способ взятия начальных условий

    ## берем из файла с готовыми НУ
    IC_fname = 'series_IC_20_20.txt'
    IC_index = 1
    IC_arr, w_arrs = mem.read_series_IC(s.temporary_path + IC_fname)


    # p.c = [8.5 + 0.001*i for i in range(p.k_elements)]
    # p.c = [8.5, 9.0]
    # p.c = [8.5 for i in range(p.k_elements)]
    p.c = [8.5 + 1*i for i in range(p.k_elements)]

    # p.w_arr = w_arrs[IC_index:IC_index+p.k_elements]
    # p.w_arr = [round(1.0 + 0.2*i, 3) for i in range(p.k_elements)]
    p.w_arr = [1. for i in range(p.k_elements)]
    IC = IC_arr[IC_index][:p.k_elements*s.k]
    # IC = []
    # for i in range(p.k_elements):
    #     IC.append(2)
    #     IC.append(2)
    #     IC.append(0)

    step_t_for_avg = 1

    ## берем случайные
    # IC = mem.generate_random_IC_ressler(5, 5, 1, k_elements)
    # p.w_arr = []


    ## Просто один эксперимент с какой-то связью
    # date = mem.hms_now().replace(':', '-')
    # path = s.grid_experiments_path
    # path = mem.make_dir(s.grid_experiments_path + f"{date} {p.k_elements}el a {p.a} avg {step_t_for_avg} another T")
    # path += '/'

    # print('Старт программы', f'Результаты в {path}')
    # s.toch = [1e-12, 1e-12]

    # for i in range(10):
    #     p.T = 0.1 + round(i / 10. ,1)
    #     print('T: ', p.T, '------------------')

    #     p.radius = 10.
    #     one_exp_couplings(IC, couplings=(0, 0, 0), couplings_rep=(1, 1, 0), sys_params=p,
    #                                 t_max=t_max, small_animation=True, system = 'rossler', save_dir=path, 
    #                                 step_t_for_avg=step_t_for_avg, time_skip=300, 
    #                                 suffix=f"avg {step_t_for_avg} T {p.T} ")

    #     p.radius = 4.
    #     one_exp_couplings(IC, couplings=(1, 1, 0), couplings_rep=(0, 0, 0), sys_params=p,
    #                                 t_max=t_max, small_animation=True, system = 'rossler', save_dir=path, 
    #                                 step_t_for_avg=step_t_for_avg, time_skip=300, 
    #                                 suffix=f"avg {step_t_for_avg} T {p.T} ")

    s.method = "LSODA"
    # s.method = "RK45"
    p.radius = 4.
    path = s.grid_experiments_path
    global t_step_xy_graphs, size_time_graph
    t_step_xy_graphs = 50
    size_time_graph = 200
    one_exp_couplings(IC, couplings=(0, 1, 0), couplings_rep=(0, 0, 0), sys_params=p,
                                    t_max=t_max, small_animation=False, system = 'rossler', save_dir=path, 
                                    step_t_for_avg=step_t_for_avg, time_skip=0, 
                                    suffix=f"avg omega")

    # for i in range(10):
    #     p.T = 0.1 + i * 0.1
    #     date = mem.hms_now().replace(':', '-')
    #     path = mem.make_dir(s.grid_experiments_path + f"{date} {p.k_elements}el a {p.a} avg {step_t_for_avg} T {p.T:0.2f}{' arctg' if is_simple_arctg else ''}")
    #     path += '/'

    #     print('Старт программы', f'Результаты в {path}')
    #     s.toch = [1e-12, 1e-12]
    #     p.radius = 5.
    #     one_exp_couplings(IC, couplings=(0, 0, 0), couplings_rep=(1, 1, 0), sys_params=p,
    #                                 t_max=t_max, small_animation=True, system = 'rossler', save_dir=path, 
    #                                 step_t_for_avg=step_t_for_avg, suffix=f'avg {step_t_for_avg}', time_skip=300)
    #     p.radius = 4.
    #     one_exp_couplings(IC,  couplings=(1, 1, 0), couplings_rep=(0, 0, 0), sys_params=p,
    #                                 t_max=t_max, small_animation=True, system = 'rossler', save_dir=path, 
    #                                 step_t_for_avg=step_t_for_avg, suffix=f'avg {step_t_for_avg}', time_skip=300)

def experiments2():
    p = Rossler_params()
    p.k_elements = 16
    t_max = 300
    p.a = 0.16
    p.T = 0.3
    s.toch = [1e-11, 1e-11]
    step_t_for_avg = 1

    ## берем из файла с готовыми НУ
    IC_fname = 'series_IC_20_20.txt'
    IC_index = 7
    IC_arr, w_arrs = mem.read_series_IC(s.temporary_path + IC_fname)
    IC = IC_arr[IC_index][:p.k_elements*s.k]

    s.method = "RK45"
    p.radius = 4.
    path = s.grid_experiments_path

    C_arr_default = [8.50, 8.501, 8.502, 8.503]
    w_arr_default = [0.9, 0.95, 1, 1.05]
    decart = list(product(C_arr_default, w_arr_default))
    c_arr = []
    w_arr = []
    for elem in decart:
        c_arr.append(elem[0])
        w_arr.append(elem[1])
    p.c = c_arr
    p.w_arr = w_arr

    # C_arr_default = [8.5, 8.7, 8.9, 9.1, 9.3, 9.5, 9.7, 9.9, 10.1, 10.3]
    # w_arr_default = [0.9, 0.92, 0.94, 0.96, 0.98, 1, 1.02, 1.04, 1.06, 1.08]
    # C_arr_default_2 = [9.5 for i in range(10)]
    # w_arr_default_2 = [1 for i in range(10)]
    # c_arr = C_arr_default_2 + C_arr_default
    # w_arr = w_arr_default + w_arr_default_2
    # p.c = c_arr
    # p.w_arr = w_arr
    global t_step_xy_graphs
    t_step_xy_graphs = 0.1

    one_exp_couplings(IC, couplings=(0, 1, 0), couplings_rep=(0, 0, 0), sys_params=p,
                                    t_max=t_max, small_animation=False, system = 'rossler', save_dir=path, 
                                    step_t_for_avg=step_t_for_avg, time_skip=0, 
                                    suffix=f"avg {step_t_for_avg} T {p.T} ")

def experiment_omega_ratio_dep_T():
    p = Rossler_params()
    p.k_elements = 2
    p.a = 0.16
    s.toch = [1e-11, 1e-11]
    s.method = "LSODA"
    p.radius = 4
    t_max = 1000
    p.w_arr = [1 for i in range(p.k_elements)]
    # p.w_arr = [1.01, 1.011]
    p.c = [8.5, 8.501]
    # p.c = [8.5 for i in range(p.k_elements)]

    T_arr = np.arange(0, 0.201, 0.001)

    IC = []
    for i in range(p.k_elements):
        IC.append(5)
        IC.append(5)
        IC.append(0)
    
    exp_omega_dep_T(IC, T_arr, couplings=(0, 1, 0), sys_params=p, t_max=t_max)

if __name__ == "__main__":
    # experiments1()
    experiment_vdp()
    # calc_amplityde_for_vdp()

    # experiment_omega_ratio_dep_T()

