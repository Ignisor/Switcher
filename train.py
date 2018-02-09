import os

import numpy as np

from model import Switcher
from utils import text_to_ids


if __name__ == '__main__':

    file_0 = open('data/0')
    file_1 = open('data/1')

    texts_0 = [l for l in file_0]
    texts_1 = [l for l in file_1]

    y_0 = [0 for _ in range(len(texts_0))]
    y_1 = [1 for _ in range(len(texts_1))]

    texts = texts_0 + texts_1
    encoded_texts = text_to_ids(texts)
    y = np.array(y_0 + y_1)

    data_amount = len(texts)
    divider = int(0.8 * data_amount)

    train_x = encoded_texts[:divider]
    test_x = encoded_texts[divider:]
    train_y = y[:divider]
    test_y = y[divider:]

    s = Switcher()
    s.init_model()

    s.model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1000)

    s.save_weights()
