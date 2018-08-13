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
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                  activation='relu',
                                  input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # model.summary()
    model.compile(loss=tf.contrib.keras.losses.categorical_crossentropy,
                  optimizer=tf.contrib.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def main():
    x_train, y_train = gen_next_batch(10000)
    x_test, y_test = gen_next_batch(1000)
    print("data loaded............................................................................")

    model = create_model((IMAGE_HEIGHT, IMAGE_WIDTH, 1),
                         MAX_CAPTCHA*CHAR_SET_LEN)
    model.fit(x_train, y_train,
              batch_size=128 * 2,
              epochs=12,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    # 輸出結果
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()

# model = keras.Sequential()
# keras.layers.Conv2D()
# # model.add(keras.layers.Embedding(vocab_size, 16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
