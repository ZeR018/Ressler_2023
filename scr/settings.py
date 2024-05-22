w = [0.98, 1.0, 0.94, 1.07, 1.02]
a = 0.22        # Параметры
b = 0.1         # системы
c = 8.5         #

k_elements = 5  # Число агентов
k = 3           # Число уравнений для одного агента (всегда 3)

radius = 4      # Радиус связи
min_radius = 0.00
T = 0.3         # Сила связи
tau = 1

# method = 'DOP853'
# method = 'LSODA'
method = 'RK45'
t_max = 50
k_str = 5                   # Число агентов в одной строке
k_col = 5                   # Число агентов в одном столбце

stopping_border_radius = 20
stopping_border_center = [0.0 ,0.0]

small_animation = True
full_animation = False
need_save_last_state = False
look_at_infinity = False
stopping_borded_work = False

plot_colors = [
    "blue",
    "orange",
    "green",
    "red",
    "indigo",
    "m",
    "purple",
    "gray",
    "olive",
    "pink",
    "black",
    "salmon",
    "tomato",
    "navy",
    "lime",
    "orchid",
    'crimson',
    'plum',
    'peru',
    'lightskyblue',
    'forestgreen',
    'tan',
    'coral',
    'goldenrod',
    'silver'
]

# Data paths
grid_experiments_path = './data/grid_experiments/'
temporary_path = './data/temp/'