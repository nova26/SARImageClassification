import json
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from functools import partial
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def detect_outlier(data_1, data_2, threshold=4):
    outliers = []
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    while True:
        didDelete = False
        index = 0
        for y in data_1:
            z_score = (y - mean_1) / std_1
            if any(np.abs(z_score) > threshold):
                data_1 = np.delete(data_1, index, 0)
                data_2 = np.delete(data_2, index, 0)
                didDelete = True
                break
            index += 1
        if not didDelete:
            break

    return data_1, data_2


def BuildModel():
    datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    X_train = np.full((1, 75, 75, 1), 0)
    y_train = np.full((1, 1), 0)

    count = 0

    if not os.path.isfile('./data/X_train_0.pkl'):
        with open("iceberg-classifier-challenge\\train.json\\data\\processed\\train.json") as file_in:
            data = json.load(file_in)
            for line in data:
                print("sample {0}/{1} {2}".format(len(X_train), len(data), line['is_iceberg']))

                is_iceberg = line['is_iceberg']
                is_iceberg = np.reshape(is_iceberg, (1, 1))

                band_1 = line['band_1']
                band_2 = line['band_2']
                band_1 = np.reshape(band_1, (75, 75, 1))
                band_2 = np.reshape(band_2, (75, 75, 1))

                band_1 = (band_1 + band_2) / 2
                band_1 = np.reshape(band_1, (1, 75, 75, 1))

                i = 0
                for batch in datagen.flow(band_1, batch_size=1):
                    i += 1
                    if i > 16:
                        break
                    X_train = np.concatenate((X_train, batch), axis=0)
                    y_train = np.concatenate((y_train, is_iceberg), axis=0)

                if len(X_train) > 8000:
                    X_train = np.delete(X_train, 0, 0)
                    y_train = np.delete(y_train, 0, 0)
                    pickle.dump(X_train, open('./data/X_train_{0}.pkl'.format(count), 'wb'))
                    pickle.dump(y_train, open('./data/y_train_{0}.pkl'.format(count), 'wb'))
                    count = count + 1
                    X_train = np.full((1, 75, 75, 1), 0)
                    y_train = np.full((1, 1), 0)

        X_train = np.delete(X_train, 0, 0)
        y_train = np.delete(y_train, 0, 0)
        pickle.dump(X_train, open('./data/X_train_{0}.pkl'.format(count), 'wb'))
        pickle.dump(y_train, open('./data/y_train_{0}.pkl'.format(count), 'wb'))

    print("Loading data ...")

    X_train = pickle.load(open('./data/X_train_0.pkl', 'rb'))
    y_train = pickle.load(open('./data/y_train_0.pkl', 'rb'))

    for x in range(1, 4):
        print('./data/X_train_{0}.pkl'.format(x))
        X_train_tmp = pickle.load(open('./data/X_train_{0}.pkl'.format(x), 'rb'))
        y_train_tmp = pickle.load(open('./data/y_train_{0}.pkl'.format(x), 'rb'))
        X_train = np.concatenate((X_train, X_train_tmp), axis=0)
        y_train = np.concatenate((y_train, y_train_tmp), axis=0)

    xMean = np.mean(X_train, axis=0)
    xStd = np.std(X_train, axis=0)

    X_train -= xMean
    X_train /= (xStd + 1e-8)

    fileName = 'Model_epoch{epoch:02d}_loss{val_loss:.4f}_acc{val_accuracy:.2f}.h5'

    modelCheckPoint = keras.callbacks.ModelCheckpoint(
        './model/' + fileName,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,

        save_weights_only=False,
        mode='auto',
        period=1
    )

    earlyStopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.1,
                                                  patience=10,
                                                  verbose=0,
                                                  mode='auto',
                                                  min_delta=0.0001,
                                                  cooldown=0,
                                                  min_lr=0)

    callbacks_list = [modelCheckPoint, earlyStopping, reduce_lr]

    DefaultConv2D = partial(Conv2D, kernel_size=3, activation='relu', padding="SAME")

    model = keras.models.Sequential([
        DefaultConv2D(filters=32, kernel_size=3, input_shape=[75, 75, 1]),
        keras.layers.BatchNormalization(),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        MaxPooling2D(pool_size=2),
        Dropout(rate=0.25),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        MaxPooling2D(pool_size=2),
        Dropout(rate=0.25),
        DefaultConv2D(filters=384),
        DefaultConv2D(filters=384),
        DefaultConv2D(filters=384),
        DefaultConv2D(filters=384),
        MaxPooling2D(pool_size=2),
        Dropout(rate=0.25),
        DefaultConv2D(filters=512),
        DefaultConv2D(filters=512),
        DefaultConv2D(filters=512),
        MaxPooling2D(pool_size=2),
        Dropout(rate=0.25),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dropout(0.5),
        Dense(units=32, activation='relu'),
        Dropout(0.5),
        Dense(units=16, activation='relu'),
        Dropout(0.5),
        Dense(units=1, activation='sigmoid')
    ])

    opt = keras.optimizers.SGD(lr=0.01, momentum=0.8, nesterov=False)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    history = model.fit(X_train, y_train, callbacks=callbacks_list,
                        validation_data=(X_val, y_val),
                        epochs=1000,
                        shuffle=True,
                        batch_size=225,
                        verbose=1
                        )

    pickle.dump(history.history, open('./data/history.pkl', 'wb'))

    pd.DataFrame(history.history).plot(figsize=(16, 16))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


if not os.path.isfile('./model/Model_epoch32_loss0.3293_acc0.85.h5'):
    BuildModel()
else:
    model = keras.models.load_model('./model/Model_epoch32_loss0.3293_acc0.85.h5')
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "is_iceberg"])
        with open("iceberg-classifier-challenge\\test.json\\data\\processed\\test.json") as file_in:
            data = json.load(file_in)
            for line in data:
                id = line['id']
                band_1 = line['band_1']
                band_2 = line['band_2']
                band_1 = np.reshape(band_1, (75, 75, 1))
                band_2 = np.reshape(band_2, (75, 75, 1))
                band_1 = np.reshape(band_1, (1, 75, 75, 1))
                band_2 = np.reshape(band_2, (1, 75, 75, 1))

                res1 = model.predict(band_1)[0][0]
                res2 = model.predict(band_2)[0][0]
                res = (res1 + res2) / 2
                writer.writerow([id, res])
