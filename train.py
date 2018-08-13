# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from gen_captcha import gen_next_batch

IMAGE_HEIGHT = 24
IMAGE_WIDTH = 97
MAX_CAPTCHA = 4
CHAR_SET_LEN = 10


def create_model(input_shape, num_classes):
  model = keras.Sequential()
  model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='SAME'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
  model.add(keras.layers.Dropout(0.4))

  model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='SAME'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
  model.add(keras.layers.Dropout(0.4))

  model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='SAME'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
  model.add(keras.layers.Dropout(0.4))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(2048, activation='relu'))
  model.add(keras.layers.Dropout(0.4))

  model.add(keras.layers.Dense(num_classes, activation='softmax'))

  model.compile(loss=tf.contrib.keras.losses.categorical_crossentropy, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), metrics=['accuracy'])
  return model


def main():
  x_train, y_train = gen_next_batch(12800)
  x_test, y_test = gen_next_batch(1280)

  model = create_model((IMAGE_HEIGHT, IMAGE_WIDTH, 1), MAX_CAPTCHA*CHAR_SET_LEN)
  model.fit(x_train, y_train, batch_size=64, epochs=1200, verbose=1, validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)

  # 輸出結果
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])


if __name__ == '__main__':
  main()
