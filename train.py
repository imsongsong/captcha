# coding=utf-8

# import tensorflow as tf
# print(tf.__version__)


from PIL import Image, ImageDraw, ImageFont
import os

fonts_folder = '/System/Library/Fonts/'
fonts_file_names = os.listdir(fonts_folder)


image = Image.new(mode='RGB', size=(400, 50*len(fonts_file_names)), color='#FFFFFF')
draw_table = ImageDraw.Draw(im=image)

for i, fontFile in enumerate(fonts_file_names):
  print(i, fontFile)
  try:
    draw_table.text(xy=(0, i*50), text=u'1234567890', fill='#FF0000', font=ImageFont.truetype(size=40, font=os.path.join(fonts_folder, fontFile)))
  except Exception:
    pass

# draw_table.text(xy=(0, 0), text=u'1234567890', fill='#FF0000', font=ImageFont.truetype(size=40, font="f/System/Library/Fonts/Thonburi.ttc"))
# draw_table.text(xy=(0, 50), text=u'1234567890', fill='#FF0000', font=ImageFont.truetype(size=40, font="f/System/Library/Fonts/Thonburi.ttc"))

image.show()  # 直接显示图片
image.save('a.png')  # 保存在当前路径下，格式为PNG
