import numpy as np
from matplotlib import pyplot as plt
from config import settings as s
import memory_worker as mem
import pprint
import os
import seaborn as sns


def calc_num_coming_cubes():
    # Для кубиков с длиной ребра 1
    size_matrix_x = 50
    size_matrix_y = 50
    size_matrix_z = 80

    size_matrix_x2 = int(size_matrix_x / 2.)
    size_matrix_y2 = int(size_matrix_y / 2.)

    a022_2 = '2024-06-04 11.37.56 a=022'
    a022_1 = '2024-06-04 12.09.35 a=022'
    a028_1 = '2024-06-04 12.47.04 a=028'
    a016_1 = '2024-06-04 15.16.26 a016'

    path = './data/grid_experiments/' + a028_1
    sol = mem.read_integration_data(path + '/integration_data.txt')
    xs, ys, zs, ts = sol
    size = len(ts)
    k_elements = len(xs)

    # Подбираем пределы сетки
    size_matrix_z = 10
    for agent in range(k_elements):
        agent_max_z = int(max(zs[agent]))
        agent_max_x = int(max(xs[agent]))
        agent_min_x = int(min(xs[agent]))
        agent_max_y = int(max(ys[agent]))
        agent_min_y = int(min(ys[agent]))

        if agent_max_z >= size_matrix_z:
            size_matrix_z = agent_max_z + 2

        if agent_max_x >= size_matrix_x2:
            size_matrix_x2 = agent_max_x + 1
            size_matrix_x = 2 * size_matrix_x2

        if agent_min_x <= - size_matrix_x2:
            size_matrix_x2 = abs(agent_min_x)
            size_matrix_x = 2 * size_matrix_x2

        if agent_max_y >= size_matrix_y2:
            size_matrix_y2 = agent_max_y + 1
            size_matrix_y = 2 * size_matrix_y2

        if agent_min_y <= - size_matrix_y2:
            size_matrix_y2 = abs(agent_min_y)
            size_matrix_y = 2 * size_matrix_y2

    # cubes_matrix = np.zeros((k_elements, size_matrix_z, size_matrix_y, size_matrix_x))

    # # Проходимся по всем агентам
    # for agent in range(k_elements):

    #     # [agent, x, y, z]
    #     last_cube_coords = [-1, -1, -1, -1]
    #     pre_last_cube_coords = [-2, -1, -1, -1]
    #     # Проходимся по всем точкам численного интегрирования
    #     for point in range(size):
    #         x = xs[agent][point]
    #         y = ys[agent][point]
    #         z = zs[agent][point]

    #         new_cube_coords = [agent, int(x) + size_matrix_x2 + 1, int(y) + size_matrix_y2 + 1, int(z)]

    #         # Если мы всё еще находимся в предыдущем кубе
    #         if new_cube_coords == last_cube_coords or new_cube_coords == pre_last_cube_coords:
    #             continue

    #         # Если мы в новом кубе
    #         try:
    #             cubes_matrix[agent][new_cube_coords[3]][new_cube_coords[2]][new_cube_coords[1]] += 1
    #         except IndexError:
    #             print(x, y, z, point)
    #             print(new_cube_coords)
    #             cubes_matrix[agent][new_cube_coords[3]][new_cube_coords[2]][new_cube_coords[1]] += 1

    #         pre_last_cube_coords = last_cube_coords
    #         last_cube_coords = new_cube_coords

    # # Пробуем получить гистограммы первого агента по слоям (по z)
    # hists_dir = path + '/hists'
    # try:
    #     os.mkdir(hists_dir)
    #     print('dir exist')
    # except Exception as e:
    #     print('dir is already exists')
    # for agent in range(1):
    #     for z in range(size_matrix_z):
    #         if z % 10 == 0:
    #             print(f'z = {z}')
    #         plt.figure()
    #         sns.heatmap(cubes_matrix[agent][z], cmap='BuPu',
    #                     xticklabels = np.arange(- size_matrix_x2, size_matrix_x2-1), 
    #                     yticklabels=np.arange(size_matrix_y2-1, -size_matrix_y2, -1))
    #         plt.title(f'agent={agent}, z={z}, y={y}')
    #         plt.savefig(hists_dir + f'/{agent}_{z}.png')
    #         plt.close()
    #         # for y in range(size_matrix_y):
    #             # plt.figure()
    #             # plt.hist(cubes_matrix[agent][z][y], edgecolor='darkblue')
    #             # plt.title(f'agent={agent}, z={z}, y={y}')
    #             # plt.grid()
    #             # plt.savefig(hists_dir + f'/{agent}_{z}_{y}.png')
    #             # plt.close()

    # Считаем количество ненулевых кубиков
    # agents_num_cubes = []
    # for agent in range(k_elements):

    #     sum = 0
    #     for z in range(size_matrix_z):
    #         for y in range(size_matrix_y):
    #             for x in range(size_matrix_x):
    #                 if cubes_matrix[agent][z][y][x] > 0:
    #                     sum += 1
    #     agents_num_cubes.append(sum)

    # print(agents_num_cubes)

    ################################################################## Если не смотреть на агентов отдельно ##################################################################



    cubes_matrix = np.zeros((size_matrix_z, size_matrix_y, size_matrix_x))

    # Проходимся по всем агентам
    for agent in range(k_elements):

        # [agent, x, y, z]
        last_cube_coords = [-1, -1, -1]
        pre_last_cube_coords = [-1, -1, -2]
        # Проходимся по всем точкам численного интегрирования
        for point in range(size):
            x = xs[agent][point]
            y = ys[agent][point]
            z = zs[agent][point]

            new_cube_coords = [int(x) + size_matrix_x2 + 1, int(y) + size_matrix_y2 + 1, int(z)]

            # Если мы всё еще находимся в предыдущем кубе
            if new_cube_coords == last_cube_coords or new_cube_coords == pre_last_cube_coords:
                continue

            # Если мы в новом кубе
            try:
                cubes_matrix[new_cube_coords[2]][new_cube_coords[1]][new_cube_coords[0]] += 1
            except IndexError:
                print(x, y, z, point)
                print(new_cube_coords)
                cubes_matrix[new_cube_coords[2]][new_cube_coords[1]][new_cube_coords[0]] += 1

            pre_last_cube_coords = last_cube_coords
            last_cube_coords = new_cube_coords

    # Считаем количество ненулевых кубиков

    sum = 0
    for z in range(size_matrix_z):
        for y in range(size_matrix_y):
            for x in range(size_matrix_x):
                if cubes_matrix[z][y][x] > 0:
                    sum += 1
    print(sum)


def make_graphs_without_grid():
    mem.plot_some_graph_without_grid('./data/grid_experiments/2023-10-28 15.24.36')

make_graphs_without_grid()