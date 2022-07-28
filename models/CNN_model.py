import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D


def read_image(path):
    """
    Function that reads an image from the input path and shows it as an example for an image from the data
    :param path: The path to the image
    :return:
    """
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {img_array.shape}")
    plt.imshow(img_array, cmap="gray")
    plt.show()


def generate_data(data_type):
    """
    Function that generates a Tensorflow dataset from the images in the input directory for training/testing images.
    https://keras.io/api/preprocessing/image/
    :param data_type: training or testing
    :return:
    """
    gen = ImageDataGenerator(rescale=1.0/255)
    if data_type.lower().strip() in ["train", "test"]:
        return gen.flow_from_directory(
            f"data/{data_type}",
            batch_size=1,
            shuffle=True,
            color_mode="grayscale",
            class_mode="categorical",
            target_size=(24, 24),
        )
    else:
        raise TypeError("Wrong input entered in 'generate_data' function!")


class CNN:

    def __init__(self, X, y, steps_per_epoch, val_steps, epochs=50):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.val_steps = val_steps
        self.model = None
        self.history = None

    def train_model(self):
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.history = self.model.fit(self.X,
                                      validation_data=self.y,
                                      epochs=self.epochs,
                                      steps_per_epoch=self.steps_per_epoch,
                                      validation_steps=self.val_steps)

    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(24, 24, 1)))
        self.model.add(MaxPooling2D(pool_size=(1, 1)))

        self.model.add(Conv2D(32, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(1, 1)))

        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(1, 1)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4, activation="softmax"))

        self.model.summary()

    def save_model(self, file_name):
        self.model.save(file_name + ".h5", overwrite=True)

    def plot_accuracy_and_loss(self):
        plt.plot(self.history.history['accuracy'], label='accuracy', color='blue')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.show()

        plt.plot(self.history.history['loss'], label='loss', color='red')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.show()


if __name__ == '__main__':
    directory_path = "data/train"
    classes = ['Closed_Eyes', 'Open_Eyes']
    image_path = 'data/train/Closed/_0.jpg'
    read_image(image_path)

    batch_size = 32
    X, y = generate_data("train"), generate_data("test")
    steps_per_epoch = len(X.classes) // batch_size
    val_steps = len(y.classes) // batch_size
    print(f"SPE: {steps_per_epoch} - Validation Steps: {val_steps}")

    img, labels = next(X)
    cnn = CNN(X, y, steps_per_epoch=steps_per_epoch, val_steps=val_steps, epochs=100)
    cnn.create_model()
    cnn.train_model()
    cnn.plot_accuracy_and_loss()
    cnn.save_model(file_name="models/drowsiness_detector_model2")