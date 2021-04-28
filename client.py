import time
import argparse
import numpy as np
import sys
import os

from train_model import part1_model
from mnist_util import load_mnist_data
from mnist_util import client_argument_parser
import pyhe_client


def test_network(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data(
        FLAGS.start_batch, FLAGS.batch_size)
    data = x_test.flatten("C")

    predictions1 = part1_model.predict(x_test)

    client = pyhe_client.HESealClient(
        FLAGS.hostname,
        FLAGS.port,
        FLAGS.batch_size,
        {FLAGS.tensor_name: (FLAGS.encrypt_data_str, predictions1)},
    )

    results = np.round(client.get_results(), 2)

    y_pred_reshape = np.array(results).reshape(FLAGS.batch_size, 10)
    with np.printoptions(precision=3, suppress=True):
        print(y_pred_reshape)

    y_pred = y_pred_reshape.argmax(axis=1)
    print("y_pred", y_pred)

    correct = np.sum(np.equal(y_pred, y_test.argmax(axis=1)))
    acc = correct / float(FLAGS.batch_size)
    print("correct", correct)
    print("Accuracy (batch size", FLAGS.batch_size, ") =", acc * 100.0, "%")


if __name__ == "__main__":
    FLAGS, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)

    test_network(FLAGS)