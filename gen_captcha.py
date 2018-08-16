# coding:utf-8
from PIL import Image, ImageDraw, ImageFont
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras


font_size = 28
font_folder = os.path.join('.', 'fonts')

IMAGE_HEIGHT = 24
IMAGE_WIDTH = 80
IMAGE_DEPTH = 1
MAX_CAPTCHA = 4
CHAR_SET_LEN = 10


# def tfdata_generator(images, labels, is_training, batch_size=128):
#   '''Construct a data generator using tf.Dataset'''

#   # def preprocess_fn(image, label):
#   #   '''A transformation function to preprocess raw data
#   #   into trainable input. '''
#   #   x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
#   #   y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
#   #   return x, y

#   dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#   if is_training:
#     dataset = dataset.shuffle(1000)  # depends on sample size

#   # Transform and batch data at the same time
#   dataset = dataset.apply(tf.contrib.data.map_and_batch(
#       preprocess_fn, batch_size,
#       num_parallel_batches=4,  # cpu cores
#       drop_remainder=True if is_training else False))
#   dataset = dataset.repeat()
#   dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

#   return dataset

def gen_dataset(batch_size=128):
  return tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)), tf.TensorShape((batch_size, MAX_CAPTCHA*CHAR_SET_LEN))))


def gen(batch_size=128):
  # return (gen_next_batch(batch_size))
  while True:
    yield gen_next_batch(batch_size)
  # yield(x, y)


def convert2gray(img):
  if len(img.shape) > 2:
    gray = np.mean(img, -1)
    return gray
  else:
    return img


def gen_next_batch(batch_size=128):
  batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], np.float32)
  batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN], np.int8)

  for i in range(batch_size):
    number = random.randint(0, 9999)
    image = gen_captcha("%04d" % number)

    # 转成灰度图片，因为颜色对于提取字符形状是没有意义的
    image = convert2gray(image)

    # batch_x[i, :] = image / 255
    batch_x[i, :] = (image.flatten() / 255).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    # print(tt.shape)
    # batch_x[i, :] = image

    arr = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN, dtype=np.int8)
    arr[number // 1000 % 10] = 1
    arr[10+number // 100 % 10] = 1
    arr[20+number // 10 % 10] = 1
    arr[30+number % 10] = 1
    batch_y[i, :] = arr

  return batch_x, batch_y


def gen_captcha(text):
  font_file = os.path.join(
      font_folder, random.choice(os.listdir(font_folder)))
  font = ImageFont.truetype(size=font_size, font=font_file)

  image = Image.new(mode='RGB', size=(IMAGE_WIDTH, IMAGE_HEIGHT), color='#FFFFFF')
  draw = ImageDraw.Draw(im=image)

  size = draw.textsize(text, font=font)
  offset = font.getoffset(text)

  draw.text(xy=(0, 0-offset[1] + random.randint(0, IMAGE_HEIGHT-size[1]+offset[1])), text=text, fill='#FF0000', font=font)

  # for i in range(10):
  #     draw.line(xy=[random.randint(0, 97), random.randint(
  #         0, 24), random.randint(0, 97), random.randint(0, 24)], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

  return np.asarray(image)


def main():
  number = random.randint(0, 9999)

  # arr = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN, dtype=np.int8)
  # arr[number // 1000 % 10] = 1
  # arr[10+number // 100 % 10] = 1
  # arr[20+number // 10 % 10] = 1
  # arr[30+number % 10] = 1

  # print(arr)

  image = gen_captcha("%04d" % number)
  print(image.shape)
  image = convert2gray(image)
  print(image.shape)
  # image = np.hstack((image, image, image))
  # print(image.shape)

  Image.fromarray(image).show()

  # 转成灰度图片，因为颜色对于提取字符形状是没有意义的
  # print(image[0][0])
  # image = convert2gray(image)
  # # batch_x[i, :] = image / 255
  # r = (image.flatten() / 255).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
  # print(r)

  # x = np.arange(4).reshape(-1, 1).astype('float32')
  # ds_x = tf.data.Dataset.from_tensor_slices(x).repeat().batch(4)
  # it_x = ds_x.make_one_shot_iterator()

  # y = np.arange(5, 9).reshape(-1, 1).astype('float32')
  # ds_y = tf.data.Dataset.from_tensor_slices(y).repeat().batch(4)
  # it_y = ds_y.make_one_shot_iterator()

  # input_vals = keras.layers.Input(tensor=it_x.get_next())
  # output = keras.layers.Dense(1, activation='relu')(input_vals)
  # model = keras.Model(inputs=input_vals, outputs=output)
  # model.compile('rmsprop', 'mse', target_tensors=[it_y.get_next()])
  # model.fit(steps_per_epoch=1, epochs=5, verbose=2)

  # ds = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape((128, 24, 80, 1)), tf.TensorShape((128, 40))))
  # x, y = ds.make_one_shot_iterator().get_next()
  # print(x, y)

  # value = ds.make_one_shot_iterator().get_next()
  # with tf.Session() as sess:
  #   # (sess.run(value))  # (1, array([1]))
  #   print(sess.run(value))  # (2, array([1, 1]))

  # batch_x, batch_y = gen_next_batch()

  # tf.enable_eager_execution()
  # tt = gen_dataset()
  # tt = tt.make_one_shot_iterator()
  # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  # print(tt)
  # # for x in range(10):
  # #     gen_captcha('1256')


if __name__ == '__main__':
  main()

# from captcha.image import ImageCaptcha  # pip install captcha
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import random
# import time
# import os

# # 验证码中的字符, 就不用汉字了
# number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# # alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# # ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# alphabet = []
# ALPHABET = []

# # 验证码一般都无视大小写；验证码长度4个字符


# def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
#   ''' 指定使用的验证码内容列表和长期 返回随机的验证码文本 '''
#   captcha_text = []
#   for i in range(captcha_size):
#     c = random.choice(char_set)
#     captcha_text.append(c)
#   return captcha_text


# def gen_captcha_text_and_image():
#   '''生成字符对应的验证码 '''
#   # image = ImageCaptcha(fonts=[os.path.join('fonts', 'OCRB10N.TTF')])  # 导入验证码包 生成一张空白图
#   image = ImageCaptcha(height=24, width=97, font_sizes=[25])

#   captcha_text = random_captcha_text()  # 随机一个验证码内容
#   captcha_text = ''.join(captcha_text)  # 类型转换为字符串
#   print(captcha_text)

#   captcha = image.generate(captcha_text)
#   # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

#   #rm  =  'rm '+captcha_text + '.jpg'
#   # os.system(rm)

#   captcha_image = Image.open(captcha)  # 转换为图片格式
#   captcha_image = np.array(captcha_image)  # 转换为 np数组类型

#   return captcha_text, captcha_image


# if __name__ == '__main__':
#   # 测试
#   while(1):
#     text, image = gen_captcha_text_and_image()
#     print('begin ', time.ctime(), type(image), image.size)
#     f = plt.figure()
#     ax = f.add_subplot(111)
#     ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
#     plt.imshow(image)
#     Image.fromarray(image).save(text+'.png')
#     plt.show()  # 显示，，取消注释并在30行取消写到文件的注释即可保存为文件
#     print('end ', time.ctime())

#     # print gen_captcha_text_and_image()
