from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from data_and_label_processing import *

def simple_linear_build(shape, class_no):
    print(shape)
    base_model_1 = VGG16(include_top=False, input_shape=(shape), classes=class_no)
    model_1= Sequential()
    model_1.add(base_model_1)
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
    inputs = Input(shape=(32, 32, 3))
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

def train(data_path, universal_path, mode, class_no, epoch_no):

    data, label = get_data_label_alltogether(data_path, universal_path, mode=mode)
    (X_train, y_train), (X_test, y_test) = train_test_split_custom(data, label, 4)

    model = create_res_net(4)
    model.compile(optimizer="RMSProp", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["acc"])

    model.fit(X_train, y_train, epochs=epoch_no)

    y_predict = model.predict(X_test)
    # print(np.argmax(y_predict, axis=1), np.argmax(y_test, axis=1))
    print("predict, test", np.argmax(y_predict, axis=1), y_test)
    #print('Accuracy:' , np.sum(np.argmax(y_predict, axis=1)==np.argmax(y_test, axis=1))/len(X_test))

    return model
