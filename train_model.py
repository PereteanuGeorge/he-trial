import mnist_util
import model
import sys
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.keras.losses import categorical_crossentropy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

(x_train, y_train, x_test, y_test) = mnist_util.load_mnist_data()

x = Input(
    shape=(
        28,
        28,
        1,
    ), name="input")
y = model.mnist_model(x)

mlp_model = Model(inputs=x, outputs=y)
print(mlp_model.summary())


def loss(labels, logits):
    return categorical_crossentropy(labels, logits, from_logits=True)


optimizer = SGD(learning_rate=0.008, momentum=0.9)
mlp_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

mlp_model.fit(
    x_train,
    y_train,
    epochs=2,
    validation_data=(x_test, y_test),
    verbose=1)

part1_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(4, 4),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten()
])
part1_model.layers[0].set_weights(mlp_model.layers[1].get_weights())
part1_model.layers[1].set_weights(mlp_model.layers[2].get_weights())
part1_model.layers[2].set_weights(mlp_model.layers[3].get_weights())
part1_model.layers[3].set_weights(mlp_model.layers[4].get_weights())
part1_model.layers[4].set_weights(mlp_model.layers[5].get_weights())
part1_model.summary()

# second part of initial model

test_loss, test_acc = mlp_model.evaluate(x_test, y_test, verbose=1)
print("\nTest accuracy:", test_acc)



def main(FLAGS):
    part2_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(1, 128), name="input"),
        tf.keras.layers.Dense(10, activation='softmax', name="output")
    ])
    part2_model.summary()
    part2_model.layers[0].set_weights(mlp_model.layers[6].get_weights())
    part2_model.layers[1].set_weights(mlp_model.layers[7].get_weights())

    mnist_util.save_model(
        tf.compat.v1.keras.backend.get_session(),
        ["output/BiasAdd"],
        "./models",
        "test_model",
    )

if __name__ == "__main__":
    FLAGS, unparsed = mnist_util.train_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags: ", unparsed)
        exit(1)

    main(FLAGS)
