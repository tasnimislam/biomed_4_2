import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tasnim_biomed.image_processing import *


def read_xlx_file(xlx_file):
    bld_summary = pd.read_excel(xlx_file, sheet_name=0, header=0, names=None, index_col=None, usecols=None)
    bld_summary['Blood_group'] = pd.Categorical(bld_summary['Blood_group'])
    bld_summary['Blood_group'].replace(['A_pos', 'B_pos', 'O_pos', 'AB_pos'],
                        [0, 1, 2, 3], inplace=True)
    return bld_summary

def get_desired_image_label(index, xlx_data, universal_path, file_name = 'li.bmp'):
    user_name = xlx_data.iloc[index, 0]
    data_path = universal_path + '/' + user_name + '/' + file_name
    data = image_preprocessing(data_path)
    bld_group = xlx_data.iloc[index, 1]
    return data, bld_group


def get_data_label_alltogether(xlx_file, universal_path, mode = 'ri.bmp'):
    xlx_data = read_xlx_file(xlx_file)
    data = []
    label = []
    for i in range(len(xlx_data)):
        x, y = get_desired_image_label(i, xlx_data, universal_path, mode)
        data.append(x)
        label.append(y)

    data = np.array(data)
    label = np.array(label)

    return data, label

def train_test_split_custom(X, y, class_no, test_split = 0.2, validation_split = 0.2):
    y = tf.keras.utils.to_categorical(y, num_classes=class_no)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_split, random_state=42)
    return (X_train, y_train), (X_test, y_test)