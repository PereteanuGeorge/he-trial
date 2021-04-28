import tensorflow as tf
from mnist_util import load_mnist_data
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    AveragePooling2D,
    Flatten,
    Convolution2D,
    MaxPooling2D,
    Reshape,
)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0
x_train, y_train, x_test, y_test = load_mnist_data()


def mnist_model(input):
    y = Conv2D(32, (3, 3), activation='relu', use_bias=True, input_shape=(28, 28, 1))(input)
    y = MaxPooling2D(4, 4)(y)
    y = Conv2D(32, (3, 3), use_bias=True, activation='relu')(y)
    y = MaxPooling2D(2, 2)(y)
    # y = Flatten()(y) TODO:check here
    known_shape = y.get_shape()[1:]
    size = np.prod(known_shape)
    print('size', size)

    # Using Keras model API with Flatten results in split ngraph at Flatten() or Reshape() op.
    # Use tf.reshape instead
    y = tf.reshape(y, [-1, size])
    y = Dense(128, use_bias=True, activation='relu')(y)
    y = Dense(10, use_bias=True, activation='softmax', name="output")(y)
    return y


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D(4, 4),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(training_images, training_labels, epochs=5)
#
# part1_model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D(4, 4),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten()
# ])
# part1_model.layers[0].set_weights(model.layers[0].get_weights())
# part1_model.layers[1].set_weights(model.layers[1].get_weights())
# part1_model.layers[2].set_weights(model.layers[2].get_weights())
# part1_model.layers[3].set_weights(model.layers[3].get_weights())
# part1_model.layers[4].set_weights(model.layers[4].get_weights())
# part1_model.summary()
#
# # second part of initial model
# part2_model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(1, 128)),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
# part2_model.summary()
# part2_model.layers[0].set_weights(model.layers[5].get_weights())
# part2_model.layers[1].set_weights(model.layers[6].get_weights())
# # predictions holds prediction for test set from initial model
# predictions = model.predict(test_images)
#
# # predictions1 holds output of first part when test set is used as input
# predictions1 = part1_model.predict(test_images)
#
# import numpy as np
#
# # tmp is used to transform prediction1 in a format recognizable from part2
# tmp = np.zeros((10000, 1, 128))
# for i in range(0, 10000):
#     tmp[i, :, :] = predictions1[i, :]
#
# # predictions2 holds the output of second part when the result of first part (predictions1) is used as input
# predictions2 = part2_model.predict(tmp)
#
# # check that the result of initial model is the same with the result of the two parts
# ok = 0
# for i in range(0, 10000):
#     if np.argmax(predictions[i]) != np.argmax(predictions2[i]):
#         print(i, " False")
#     else:
#         ok = ok + 1
# print(ok)
