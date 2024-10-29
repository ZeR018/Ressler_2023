from default_model import func_rossler_2_dim_params_maker
from scipy.integrate import solve_ivp
import settings as s
from matplotlib import pyplot as plt
import memory_worker as mem
import numpy as np
import time
import joblib
from matplotlib.animation import ArtistAnimation


def one_exp_couplings(IC, w_arr, a, isSolo = True, couplings = (False, True, False), 
                      k_elements = s.k_elements, t_max = s.t_max, tau = s.tau, small_animation = True):

    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())

    func_rossler_2_dim_params = func_rossler_2_dim_params_maker(k_elements, couplings)
    sol = solve_ivp(func_rossler_2_dim_params, [0, t_max], IC, args=(w_arr, a, tau), 
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

    plot_colors = mem.make_colors(k_elements)

    if isSolo:

        suffix = f"coup_{'x' if couplings[0] else ''}{'y' if couplings[1] else ''}{'z' if couplings[2] else ''}"

        path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, w_arr, [], [], k_elements=k_elements, a=a, tau=tau, dir_name_suffix=suffix)
        plt.close()
        lim_param = 20
        mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 100, 
                                               mashtab=[-lim_param, lim_param, -lim_param, lim_param])

        # Анимация y(x)
        if small_animation:
            frames, fig_gif = mem.make_frames_grid_agents(xs, ys, plot_colors, frames_step=100, _k_elements = k_elements, lims=[-lim_param, lim_param, -lim_param, lim_param])
            interval = 120
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

        plt.figure(figsize=[10, 6])
        start = np.searchsorted(ts, 50, side='left')
        end = np.searchsorted(ts, 100, side="left")
        for agent in range(k_elements):
            plt.plot(ts[start:end], xs[agent][start:end], color=plot_colors[agent], label=f'agent {agent+1}')
        plt.grid()
        plt.ylim(-20, 20)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        plt.savefig(path_save + '/xt_final.png')

    return 0

k_elements = 10
a = 0.22
IC_fname = 'series_IC_20_20.txt'
IC_index = 0
IC_arr, w_arr = mem.read_series_IC(s.temporary_path + IC_fname)
IC = IC_arr[IC_index]

one_exp_couplings(IC, w_arr, a, couplings=(0, 0, 1), k_elements=k_elements, t_max=1000, tau=1, small_animation=False)