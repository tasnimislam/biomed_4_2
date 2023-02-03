from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from data_and_label_processing import *
from predict import *
from sklearn.metrics import plot_roc_curve, auc

from sklearn.ensemble import RandomForestClassifier


def simple_linear_build(shape, class_no):
    # print(shape)
    # base_model_1 = VGG16(include_top=False, input_shape=(shape), classes=class_no)
    model_1= Sequential()
    # model_1.add(base_model_1)
    model_1.add(Flatten())
    model_1.add(Dense(1024,activation=('relu'),input_dim=512))
    model_1.add(Dense(512,activation=('relu')))
    model_1.add(Dense(256,activation=('relu')))
    model_1.add(Dense(128,activation=('relu')))
    model_1.add(Dense(class_no,activation=('softmax'))) #This is the classification layer
    return model_1


def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x, downsample, filters, kernel_size = 3):
    y = Conv2D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def create_res_net(class_no):
    inputs = Input(shape=(128, 32, 3))
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(class_no, activation='softmax')(t)

    model = Model(inputs, outputs)

    return model

def train(data_path, universal_path, class_no, epoch_no, class_used, run_number):

    data, label = get_data_label_alltogether(data_path, universal_path, class_used)
    (X_train, y_train), (X_test, y_test) = train_test_split_custom(data, label, class_no)

    model = simple_linear_build((32, 32, 3), class_no)
    model.compile(optimizer="RMSProp", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["acc"])

    model.fit(X_train, y_train, epochs=epoch_no, batch_size = 8)
    values, counts = np.unique(label, return_counts=True)
    predict(X_test, y_test, model, class_used, run_number, counts)

    return model

def train_sklearn(data_path, universal_path, class_no, epoch_no, class_used, run_number):
    data, label = get_data_label_alltogether(data_path, universal_path, class_used)
    (X_train, y_train), (X_test, y_test) = train_test_split_custom(data, label, class_no)
    print(X_train.shape)
    X_train = np.reshape(X_train, (-1, 4*288*256))
    X_test = np.reshape(X_test, (-1, 4*288*256))

    model = RandomForestClassifier().fit(X_train, y_train)
    values, counts = np.unique(label, return_counts=True)
    predict(X_test, y_test, model, class_used, run_number, counts)