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

img_name = 'two_lorenz_sync'

img = Image.open(f'./data/saved_graphs/{img_name}.png')

# Создаем объект для рисования
draw = ImageDraw.Draw(img)

# Указываем шрифт и размер (по умолчанию используется стандартный шрифт)
font = ImageFont.truetype(font_name, size=48)

# Добавляем текст на изображение
# draw.text((115, 180), "(a)", fill="black", font=font)
# draw.text((750, 180), "(b)", fill="black", font=font)

draw.text((20, 240), "a", fill="black", font=font)
draw.text((620, 240), "b", fill="black", font=font)

# Сохраняем измененное изображение

img.show()  # Показываем изображение
img.save(f'./data/saved_graphs/new/{img_name}_new2.png')

