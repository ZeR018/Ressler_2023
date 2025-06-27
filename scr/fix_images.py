from PIL import ImageDraw, ImageFont, Image

symbols = ['a', 'b', 'c', 'd', 'e', 'f']
font_name = "./data/temp/times-new-roman-italic.ttf"

################### attractors #########################################################
# index = 0

# img_names = ['rossler_a016_yx', 'rossler_a022_yx', 'lorenz_default_zx', 'lorenz_default_zu', 'lorenz_intermittent_yx']

# img_name = img_names[index]

# img = Image.open(f'./data/saved_graphs/{img_name}.png')

# # Создаем объект для рисования
# draw = ImageDraw.Draw(img)

# # Указываем шрифт и размер (по умолчанию используется стандартный шрифт)
# font = ImageFont.truetype(font_name, size=48)

# # Добавляем текст на изображение
# # draw.text((190, 270), f"{symbols[index]}", fill="black", font=font)
# draw.text((20, 340), f"{symbols[index]}", fill="black", font=font)

# # Сохраняем измененное изображение

# img.show()  # Показываем изображение
# # img.save(f'./data/saved_graphs/{img_name}_new.png')
# img.save(f'./data/saved_graphs/new/{img_name}_new2.png')

################### coup_xy #########################################################
# index = 2

# img_names = ['coup_y', 'coup_x', 'coup_grid']
# img_name = img_names[index]

# img = Image.open(f'./data/saved_graphs/{img_name}.png')

# # Создаем объект для рисования
# draw = ImageDraw.Draw(img)

# # Указываем шрифт и размер (по умолчанию используется стандартный шрифт)
# font = ImageFont.truetype(font_name, size=48)

# # Добавляем текст на изображение
# # draw.text((130, 465), f"({symbols[index]})", fill="black", font=font)
# draw.text((20, 540), f"{symbols[index]}", fill="black", font=font)

# # Сохраняем измененное изображение

# img.show()  # Показываем изображение
# img.save(f'./data/saved_graphs/new/{img_name}_new2.png')

################################# two lorenz ################################

# img_name = 'two_lorenz_sync'

# img = Image.open(f'./data/saved_graphs/{img_name}.png')

# # Создаем объект для рисования
# draw = ImageDraw.Draw(img)

# # Указываем шрифт и размер (по умолчанию используется стандартный шрифт)
# font = ImageFont.truetype(font_name, size=48)

# # Добавляем текст на изображение
# # draw.text((115, 180), "(a)", fill="black", font=font)
# # draw.text((750, 180), "(b)", fill="black", font=font)

# draw.text((20, 240), "a", fill="black", font=font)
# draw.text((620, 240), "b", fill="black", font=font)

# # Сохраняем измененное изображение

# img.show()  # Показываем изображение
# img.save(f'./data/saved_graphs/new/{img_name}_new2.png')

################################# timeline graphs ################################

# img_names = ['As_avg_diff_11', 'phases_avg_diff_yx_11']
# # img_names = ['parall_As_diff', 'parall_phases_diff_yx_1']
# # img_names = ['As_diff', 'phases_diff_yx_1']
# index = 0

# img = Image.open(f'./data/saved_graphs/{img_names[index]}.png')

# # Создаем объект для рисования
# draw = ImageDraw.Draw(img)

# # Указываем шрифт и размер (по умолчанию используется стандартный шрифт)
# font = ImageFont.truetype(font_name, size=48)

# # Добавляем текст на изображение
# # draw.text((115, 180), "(a)", fill="black", font=font)
# # draw.text((750, 180), "(b)", fill="black", font=font)

# draw.text((25, 235), symbols[index], fill="black", font=font)
# # draw.text((10, 100), symbols[index], fill="black", font=font)


# # Сохраняем измененное изображение

# img.show()  # Показываем изображение
# img.save(f'./data/saved_graphs/new/{img_names[index]}_new.png')


################################# xy graphs ################################

# img_names = ['omega_dep_t_2', 'omega_dep_t_10', 'omegas_dep_T']
# img_names = ['posl_y_c_10', 'parallel_y_c_10_2', 'grid_c_omega_10 (2)']
# img_names = ['posl_xy', 'parallel_xy_c']
# img_names = ['posl_xy_c_10', 'parallel_xy_c_10', "grid_c_omega_10"]
# img_names = ['a_dep_a', 'a_dep_c', 'omega_dep_a', 'omega_dep_c']
# img_names = ['rossler_16_vdp_1', 'rossler_16_vdp_2', 'vdp_amplitudes']
# img_names = ['vdp_posl', 'vdp_parallel_4']
# img_names = ['vdp_posl_2', 'vdp_parallel_2']
# img_names = ['rossler_a016_yx', 'rossler_a022_yx', 'lorenz_default_zx', 'lorenz_default_zu', 'lorenz_intermittent_yx']
# img_names = ['oscillatory_death_xt', 'oscillatory_death_yt']
img_names = ['vdp_sync_rossler_xt', 'vdp_sync_rossler_yt']
symbols = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
# index = 1
# img = Image.open(f'./data/saved_graphs/{img_names[index]}.png')
# # Создаем объект для рисования
# draw = ImageDraw.Draw(img)

# # Указываем шрифт и размер (по умолчанию используется стандартный шрифт)
# font = ImageFont.truetype(font_name, size=48)

# # Добавляем текст на изображение
# # draw.text((30, 415), symbols[index], fill="black", font=font)
# draw.text((30, 335), symbols[index], fill="black", font=font)


# # Сохраняем измененное изображение
# img.show()  # Показываем изображение
# img.save(f'./data/saved_graphs/new/{img_names[index]}_new.png')

for index in range(len(img_names)):
    img = Image.open(f'./data/saved_graphs/{img_names[index]}.png')
    # Создаем объект для рисования
    draw = ImageDraw.Draw(img)

    # Указываем шрифт и размер (по умолчанию используется стандартный шрифт)
    font = ImageFont.truetype(font_name, size=48)

    # Добавляем текст на изображение
    # draw.text((30, 415), symbols[index], fill="black", font=font)
    draw.text((30, 335), symbols[index], fill="black", font=font)


    # Сохраняем измененное изображение
    img.show()  # Показываем изображение
    img.save(f'./data/saved_graphs/new/{img_names[index]}_new.png')