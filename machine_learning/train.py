import data
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def train_addition(train_dataset, test_dataset, train_data, test_data, normalizer):
    # get addition results from datasets
    add_target = train_dataset.get_y_add()
    test_add_target = test_dataset.get_y_add()

    # addition model follows linear regression
    add_model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    add_model.summary()

    loss = tf.keras.losses.MeanAbsoluteError()
    optim = tf.keras.optimizers.Adam(lr=0.1)

    add_model.compile(optimizer=optim, loss=loss)

    add_model.fit(
        train_data,
        add_target,
        epochs=120,
        verbose=1,
        validation_data=(test_data, test_add_target)
    )

    # add_model.save('models/addition_model')

def train_subtraction(train_dataset, test_dataset, train_data, test_data, normalizer):
    # get subtraction results from datasets
    sub_target = train_dataset.get_y_sub()
    test_sub_target = test_dataset.get_y_sub()

    # subtraction model follows linear regression
    sub_model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    sub_model.summary()

    loss = tf.keras.losses.MeanAbsoluteError()
    optim = tf.keras.optimizers.Adam(lr=0.1)

    sub_model.compile(optimizer=optim, loss=loss)

    sub_model.fit(
        train_data,
        sub_target,
        epochs=120,
        verbose=1,
        validation_data=(test_data, test_sub_target)
    )

    # sub_model.save('models/subtraction_model')

def train_multiplication(train_dataset, test_dataset, train_data, test_data, normalizer):
    # get multiplication results from datasets
    mul_target = train_dataset.get_y_mul()
    test_mul_target = test_dataset.get_y_mul()

    mul_model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(units=200, input_dim=1),
        layers.Activation('relu'),
        layers.Dense(units=45),
        layers.Activation('relu'),
        layers.Dense(units=1)
    ])
    mul_model.summary()

    loss = tf.keras.losses.MeanAbsoluteError()
    optim = tf.keras.optimizers.Adam(lr=0.01)

    mul_model.compile(optimizer=optim, loss=loss)

    mul_model.fit(
        train_data,
        mul_target,
        epochs=1000,
        verbose=1,
        validation_data=(test_data, test_mul_target)
    )


def predict(model):
    prediction = model.predict(np.array([[2,2],[500, 200],[-32, 32],[100000, 5000000]]))
    f = open("prediction.txt", "w")
    for i in prediction:
        f.write(i[0].astype('str') + '\n')
    f.close()

if __name__ == "__main__":
    #train_dataset = data.Data()
    #train_dataset.generate_data(10000, 20000)

    # load training dataset and get x values
    train_dataset = data.Data('datasets/train_dataset.npy')
    train_data = train_dataset.get_x()
    # load testing dataset and get x values
    test_dataset = data.Data('datasets/test_dataset.npy')
    test_data = test_dataset.get_x()

    # adapt normalizer to x values
    normalizer = preprocessing.Normalization()
    normalizer.adapt(train_data)

    #train_addition(train_dataset, test_dataset, train_data, test_data, normalizer)
    train_subtraction(train_dataset, test_dataset, train_data, test_data, normalizer)
    #train_multiplication(train_dataset, test_dataset, train_data, test_data, normalizer)