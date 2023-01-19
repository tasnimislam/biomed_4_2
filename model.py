from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tasnim_biomed.data_and_label_processing import *

def build(shape, class_no):
    model_1= Sequential()
    model_1.add(Flatten())
    model_1.add(Dense(1024,activation=('relu'),input_dim=512))
    model_1.add(Dense(512,activation=('relu')))
    model_1.add(Dense(256,activation=('relu')))
    model_1.add(Dense(128,activation=('relu')))
    model_1.add(Dense(class_no,activation=('softmax'))) #This is the classification layer
    return model_1

def train(data_path, universal_path, mode, class_no, epoch_no):

    data, label = get_data_label_alltogether(data_path, universal_path, mode=mode)
    (X_train, y_train), (X_test, y_test) = train_test_split_custom(data, label, 4)

    model = build(shape=(None,X_train.shape[1], X_train.shape[2]), class_no=class_no)
    model.compile(optimizer="Adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

    model.fit(X_train, y_train, epochs=epoch_no)

    y_predict = model.predict(X_test)
    print(np.argmax(y_predict, axis=1), np.argmax(y_test, axis=1))
    print('Accuracy:' , np.sum(np.argmax(y_predict, axis=1)==np.argmax(y_test, axis=1))/len(X_test))

    return model
