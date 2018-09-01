# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from gen_captcha import gen_next_batch, gen_dataset

IMAGE_HEIGHT = 24
IMAGE_WIDTH = 80
IMAGE_DEPTH = 1
MAX_CAPTCHA = 4
CHAR_SET_LEN = 10


def create_model(input_shape, num_classes):
  model = keras.Sequential()
  model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='SAME'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
  # model.add(keras.layers.Dropout(0.95))

  model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='SAME'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
  # model.add(keras.layers.Dropout(0.95))

  model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='SAME'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
  model.add(keras.layers.Dropout(0.05))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(1024, activation='relu'))
  # model.add(keras.layers.Dropout(0.05))

  model.add(keras.layers.Dense(num_classes, activation='sigmoid'))
  model.compile('adadelta', loss='binary_crossentropy', metrics=['accuracy'])
  # model.compile(tf.train.AdamOptimizer(learning_rate=0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
  # model.compile(tf.train.AdamOptimizer(learning_rate=0.001), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

  from keras.utils.vis_utils import plot_model
  plot_model(model, to_file='Model/model.png', show_shapes=True)

  return model


def main():
  model = create_model((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), MAX_CAPTCHA * CHAR_SET_LEN)

  # x_train, y_train = gen_next_batch(12800)
  # x_val, y_val = gen_next_batch(128)
  # model.fit(x_train, y_train, batch_size=64, epochs=1200, verbose=1, validation_data=(x_val, y_val))

  x_train, y_train = gen_dataset(64).make_one_shot_iterator().get_next()
  # x_val, y_val = gen_dataset().make_one_shot_iterator().get_next()
  model.fit(x_train, y_train, steps_per_epoch=50, epochs=1000, verbose=1)

  x_test, y_test = gen_next_batch(128)
  score = model.evaluate(x_test, y_test, verbose=0)

  # 輸出結果
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])


if __name__ == '__main__':
  main()
