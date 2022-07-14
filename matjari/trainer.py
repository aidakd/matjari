from matjari.load_data import get_data, load_json, load_img, encod, split_data
from matjari.load_data import get_data_using_blob
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import pandas as pd
import numpy as np
import joblib
from google.cloud import storage

BUCKET_NAME='wagon-data-839-melliani'
STORAGE_LOCATION = 'models/model_cnn.joblib'

def get_model(X, y):
    #build pipeline later
    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(256, 256, 3)))

    model.add(layers.Conv2D(16, (10,10), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(3,3)))

    model.add(layers.Conv2D(32, (8,8), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(3,3)))

    model.add(layers.Conv2D(32, (6,6), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(3,3)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(15, activation='softmax'))
    #compile
    learning_rate=1e-4
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    print('model compiled')
    es = EarlyStopping(patience = 20, restore_best_weights = True)
    model.fit(X, y,
            batch_size=16,
            epochs=2,
            validation_split=0.3,
            callbacks=[es],
            verbose = 0)
    print('model fitted')
    return model

# def fit_model(self, X, y):
#     es = EarlyStopping(patience = 20, restore_best_weights = True)
#     #data = get_data()
#     #data = load_json(data)
#     #data = load_img(data)
#     #X, y = encod(data)
#     #X_train, X_test, y_train, y_test = split_data(X, y)
#     self.fit(X, y,
#              batch_size=None,
#              epochs=2,
#              validation_split=0.3,
#              callbacks=[es],
#              verbose = 2)
#     print('model fitted')
#     return self

def evaluate_model(model, X, y):
    #data = get_data()
    #data = load_json(data)
    #data = load_img(data)
    #X, y = encod(data)
    #X_train, X_test, y_train, y_test = split_data(X, y)
    score = model.evaluate(X, y, verbose=2)
    print('evaluating done')
    return score

def predict_model(model, X_new):
    y_pred = model.predict(X_new)
    return y_pred

def save_model(model):
    joblib.dump(model, 'model.joblib')
    print('model saved')

def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model.joblib')
    print("uploaded model.joblib to gcp cloud storage!")


if __name__ == "__main__":
    #what to test
    df = get_data()
    get_data_using_blob()
    df_merge = load_json(df)
    df_merge = load_img(df_merge)
    X, y = encod(df_merge)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = get_model(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    save_model(model)
    upload_model_to_gcp()
