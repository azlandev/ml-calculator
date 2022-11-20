import data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def train_addition(train_dataset, test_dataset, train_data, test_data):
    # get addition results from datasets
    add_target = train_dataset.get_y_add()
    test_add_target = test_dataset.get_y_add()

    # adapt normalizer to x values
    normalizer = preprocessing.Normalization()
    normalizer.adapt(train_data)

    # addition model follows linear regression
    add_model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    add_model.summary()

    loss = tf.keras.losses.MeanAbsoluteError()
    optim = tf.keras.optimizers.Adam(learning_rate=0.1)

    add_model.compile(optimizer=optim, loss=loss)

    add_model.fit(
        train_data,
        add_target,
        epochs=120,
        verbose=1,
        validation_data=(test_data, test_add_target)
    )

    # add_model.save('models/addition_model')

def train_subtraction(train_dataset, test_dataset, train_data, test_data):
    # get subtraction results from datasets
    sub_target = train_dataset.get_y_sub()
    test_sub_target = test_dataset.get_y_sub()

    # adapt normalizer to x values
    normalizer = preprocessing.Normalization()
    normalizer.adapt(train_data)

    # subtraction model follows linear regression
    sub_model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    sub_model.summary()

    loss = tf.keras.losses.MeanAbsoluteError()
    optim = tf.keras.optimizers.Adam(learning_rate=0.1)

    sub_model.compile(optimizer=optim, loss=loss)

    sub_model.fit(
        train_data,
        sub_target,
        epochs=120,
        verbose=1,
        validation_data=(test_data, test_sub_target)
    )

    # sub_model.save('models/subtraction_model')

def train_multiplication(train_dataset, test_dataset, train_data, test_data):
    # get multiplication results from datasets
    mul_target = np.abs(train_dataset.get_y_mul())
    test_mul_target = np.abs(test_dataset.get_y_mul())

    # convert data to positive
    train_data = np.abs(train_data)
    test_data = np.abs(test_data)

    # normalize data using log normalization
    train_normalized = np.log(train_data)
    test_normalized = np.log(test_data)
    target_normalized = np.log(mul_target)
    test_target_normalized = np.log(test_mul_target)
    normalizer = preprocessing.Normalization()
    normalizer.adapt(train_normalized)

    mul_model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    mul_model.summary()

    loss = tf.keras.losses.MeanAbsoluteError()
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)

    mul_model.compile(optimizer=optim, loss=loss)

    mul_model.fit(
        train_normalized,
        target_normalized,
        epochs=50,
        verbose=1,
        validation_data=(test_normalized, test_target_normalized)
    )

    # mul_model.save('models/multiplication_model')

def train_division(train_dataset, test_dataset, train_data, test_data):
    # get division results from datasets
    div_target = np.abs(train_dataset.get_y_div())
    test_div_target = np.abs(test_dataset.get_y_div())

    # convert data to positive
    train_data = np.abs(train_data)
    test_data = np.abs(test_data)

    # normalize data using log normalization
    train_normalized = np.log(train_data)
    test_normalized = np.log(test_data)
    target_normalized = np.log(div_target)
    test_target_normalized = np.log(test_div_target)
    normalizer = preprocessing.Normalization()
    normalizer.adapt(train_normalized)

    div_model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    div_model.summary()

    loss = tf.keras.losses.MeanAbsoluteError()
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)

    div_model.compile(optimizer=optim, loss=loss)

    div_model.fit(
        train_normalized,
        target_normalized,
        epochs=20,
        verbose=1,
        validation_data=(test_normalized, test_target_normalized)
    )

    div_model.save('models/division_model')

if __name__ == "__main__":
    #train_dataset = data.Data()
    #train_dataset.generate_data(10000, 20000)

    # load training dataset and get x values
    train_dataset = data.Data('datasets/train_dataset.npy')
    train_data = train_dataset.get_x()
    # load testing dataset and get x values
    test_dataset = data.Data('datasets/test_dataset.npy')
    test_data = test_dataset.get_x()

    #train_addition(train_dataset, test_dataset, train_data, test_data)
    #train_subtraction(train_dataset, test_dataset, train_data, test_data)
    #train_multiplication(train_dataset, test_dataset, train_data, test_data)
    train_division(train_dataset, test_dataset, train_data, test_data)