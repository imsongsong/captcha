from PIL import Image, ImageDraw, ImageFont
import os

# fonts_folder = 'C:\\Windows\\Fonts'
fonts_folder = "./fonts"
fonts_file_names = os.listdir(fonts_folder)


image = Image.new(mode='RGB', size=(
    800, 28*len(fonts_file_names)), color='#FFFFFF')
draw = ImageDraw.Draw(im=image)

for i, fontFile in enumerate(fonts_file_names):
    print(i, fontFile)
    try:
        font = ImageFont.truetype(
            size=28, font=os.path.join(fonts_folder, fontFile))
        text = u'1234567890'

        draw.line([0, i*28, 800, i*28], fill=(255, 100, 255))
        size = draw.textsize(text, font=font)

        offset = font.getoffset(text)
        print(size[1]-offset[1])

        draw.text(xy=(0, i*28-offset[1]), text=u'1234567890   '+fontFile, fill='#FF0000',
                  font=font)

        # print(size, offset)

    except Exception:
        pass

# draw.text(xy=(0, 0), text=u'1234567890', fill='#FF0000', font=ImageFont.truetype(size=40, font="f/System/Library/Fonts/Thonburi.ttc"))
# draw.text(xy=(0, 50), text=u'1234567890', fill='#FF0000', font=ImageFont.truetype(size=40, font="f/System/Library/Fonts/Thonburi.ttc"))

image.show()  # 直接显示图片
image.save('d.png')  # 保存在当前路径下，格式为PNGsP
