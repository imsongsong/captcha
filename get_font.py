import os
import shutil

system_font_folder = 'C:\\Windows\\Fonts'
local_font_folder = 'C:\\Users\\Songsong\\Desktop\\captcha\\fonts'

fonts = [
    'arial.ttf',
    'ebrima.ttf',
    'gadugi.ttf',
    'leelawad.ttf',
    'leelawUI.ttf',
    'micross.ttf',
    'seguiemj.ttf',
    'seguisym.ttf',
    'calibri.ttf',
    'framdcn.ttf',
    'lsans.ttf',
    'lucon.ttf',
    'malgun.ttf',
    'rock.ttf',
    'simhei.ttf',
    'taile.ttf',
    'trebuc.ttf',
    'verdana.ttf'
]

for font in fonts:
    shutil.copyfile(os.path.join(system_font_folder, font),
                    os.path.join(local_font_folder, font))
