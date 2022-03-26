import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def read_image(path):
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {img_array.shape}")
    plt.imshow(img_array, cmap="gray")
    plt.show()


# Transform images to grayscale and to fixed size of 224
def transform_images(directory, classes):
    img_size = 224
    for category in classes:
        path = os.path.join(directory, category)
        for img in os.listdir(path):
            # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            new_image = cv2.resize(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE), (img_size, img_size))
            # plt.imshow(new_image, cmap="gray")
            # plt.show()
            # break


def create_training_data(directory, classes):
    training_data = []
    img_size = 224
    for category in classes:
        path = os.path.join(directory, category)
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img = cv2.resize(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE), (img_size, img_size))
                training_data.append([img, class_num])

            except Exception as e:
                pass

    return training_data


def create_model(X, y):
    model = tf.keras.applications.mobilenet.MobileNet()
    print(f"Model Summary:\n{model.summary()}")

    base_input = model.layers[0].input
    base_output = model.layers[-4].output
    Flat_layer = layers.Flatten()(base_output)
    final_output = layers.Dense(1)(Flat_layer)
    final_output = layers.Activation('sigmoid')(final_output)

    new_model = keras.Model(inputs=base_input, outputs=final_output)
    print(f"New Model Summary:\n{new_model.summary()}")

    new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    new_model.fit(X, y, epochs=10, validation_split=0.1)
    # new_model.save('drowsiness_detector_model.h5')


def test_model(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(image, (224, 224))


if __name__ == '__main__':
    directory_path = "data/train"
    classes = ['Closed_Eyes', 'Open_Eyes']
    # image_path = 'data/train/Closed_Eyes/s0001_00001_0_0_0_0_0_01.png'
    # read_image(image_path)

    # transform_images(directory_path, classes)
    training_data = create_training_data(directory_path, classes)
    print(len(training_data))
    random.shuffle(training_data)
    X, y = list(), list()
    for features, label in training_data:
        X.append(features)
        y.append(label)

    create_model(X, y)
