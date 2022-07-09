import pandas as pd
import matplotlib.pyplot as plt
import os, json
import pandas as pd
from PIL import Image
import numpy as np
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from google.cloud import storage

path_to_data = 'gs://wagon-data-839-melliani/data/matjari-dataset-cleaned.csv'
#local_path = 'raw_data/matjari-dataset-cleaned.csv'

BUCKET_NAME='wagon-data-839-melliani'
folder="data/"
path_jsons_jpg_files =  'raw_data/'+folder.replace('/','')
#path_jsons_jpg_files = 'raw_data/pictures-from-script/ALL Pictures'

def get_data():
    df = pd.read_csv(path_to_data, sep='\t')
    df.columns = ['Product Label','url']
    return df
#get_data()
def get_data_using_blob():

    # Create the folder locally
    if not os.path.exists('raw_data/'+folder):
        os.makedirs('raw_data/'+folder)

    # Retrieve all blobs with a prefix matching the folder
    client = storage.Client()
    bucket=client.get_bucket(BUCKET_NAME)
    blobs=list(bucket.list_blobs(prefix=folder))
    for blob in blobs:
        if(not blob.name.endswith("/")):
            blob.download_to_filename('raw_data/'+blob.name)
    print('function blob succed')

#get_data_using_blob()

def load_json(df):

    json_files = [pos_json for pos_json in os.listdir(path_jsons_jpg_files) if pos_json.endswith('.json')]
    jsons_data = pd.DataFrame(columns=["url", "key", "status", "error_message", "width", "height", "original_width", "original_height", "exif", "md5"])

    for index, js in enumerate(json_files):
        with open(os.path.join(path_jsons_jpg_files, js)) as json_file:
            json_text = json.load(json_file)

            # here you need to know the layout of your json and each json has to have
            # the same structure
            url = json_text["url"]
            key = json_text["key"]
            status = json_text["status"]
            error_message = json_text["error_message"]
            width = json_text["width"]
            height = json_text["height"]
            original_width = json_text["original_width"]
            original_height = json_text["original_height"]
            exif = json_text["exif"]
            md5 = json_text["md5"]
            # here I push a list of data into a pandas DataFrame at row given by 'index'
            jsons_data.loc[index] = [url, key, status, error_message, width, height, original_width, original_height, exif, md5]
            df2 = pd.DataFrame(jsons_data)
    df_merge = df.merge(df2, how='inner', on='url')
    print('function json succed')
    return df_merge

#df = get_data()
#df_merge = load_json(df)

def load_img(df_merge):
    A = []
    for i in range(0, len(df_merge)):
        img = Image.open(path_jsons_jpg_files+'/' + str(df_merge['key'][i]) + '.jpg')
        img_array = asarray(img)
        A.append(img_array)
    df_merge['A'] = A
    print('function img succed')
    return df_merge

#def_merge = load_img(df_merge)

#this function was just a step before uploading data and model to GCP to run data and model locally
#def load_nrows(df_merge):
#    nrows = 100
#    df_merge = df_merge.sample(n=nrows, random_state = 11)
#    return df_merge

def encod(df_merge):
    encoder = LabelEncoder()
    df_merge['Label_encoded'] = encoder.fit_transform(df_merge['Product Label'])
    y_encod = df_merge['Label_encoded']
    y = to_categorical(y_encod, num_classes=y_encod.nunique())
    X = np.array(df_merge['A'].tolist())
    print('function encod succed')
    return X, y

#X, y = encod(df_merge)

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print('function split succed')
    return X_train, X_test, y_train, y_test

#split_data(X,y)

if __name__ == '__main__':
    df = get_data()
    get_data_using_blob()
    df_merge = load_json(df)
    df_merge = load_img(df_merge)
    X, y = encod(df_merge)
    X_train, X_test, y_train, y_test = split_data(X, y)
