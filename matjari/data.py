import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from google.cloud import storage

#pkl to be uploaded to GCP
path_to_data = 'gs://wagon-data-839-melliani/datademo/final_data_set.pkl'
#path_to_data = '/Users/Safaemichelot/code/aidakd/matjari/raw_data/final_data_set.pkl'

#BUCKET_NAME='wagon-data-839-melliani'
#folder="datademo/"

def get_final_data():
    df = pd.read_pickle(path_to_data)
    data= df.reset_index()
    list_X = []
    list_y = []
    id_to_drop =[]
    for i, v in enumerate(data['X'][0:14999]):
        if v.shape==(256,256,3):
            list_X.append(v)
        else:
            id_to_drop.append(i)
        if i not in id_to_drop:
            list_y.append(data['Product Label'][i])
    X = np.array(list_X)
    y = list_y
    print('function getting data succed')
    return X, y

def encod(y):
    encoder = LabelEncoder()
    y_encod = encoder.fit_transform(y)
    y = to_categorical(y_encod, num_classes=len(np.unique(y_encod)))
    print('function encod succed')
    return y

#X, y = encod(df_merge)

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print('function split succed')
    return X_train, X_test, y_train, y_test

#split_data(X,y)

if __name__ == '__main__':
    X, y = get_final_data()
    y = encod(y)
    X_train, X_test, y_train, y_test = split_data(X, y)
