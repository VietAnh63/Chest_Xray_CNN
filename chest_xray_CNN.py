from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imutils import paths
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from keras.utils import np_utils
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dense,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import random
import os
import cv2, h5py
import time


verbose = 0
path_files = 'chest_xray/'
img_dims = 224
batch_size = 32
epochs = 100
image_path_train = list(paths.list_images(path_files + 'train'))
random.shuffle(image_path_train)
name = "chest_xray-{}".format(int(time.time()))
# Đường dẫn ảnh 
labels_train = [p.split(os.path.sep)[-2] for p in image_path_train]

# Chuyển tên folder thành số
le_train = LabelEncoder()
labels_train = le_train.fit_transform(labels_train)
steps_per_epoch = labels_train.shape[0] / batch_size 


def build_model():
        baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_dims, img_dims, 3)))
        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in baseModel.layers])
        #x = baseModel.layers[10].output
        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block3_conv3'].output
        # First conv block
        
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D()(x)

        # Second conv block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        # Third conv block
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        # Fifth conv block
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Dropout(rate=0.2)(x)

        # Sixth conv block
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Dropout(rate=0.2)(x)

        # FC layer
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(rate=0.7)(x)


        # Output layer
        output = Dense(units=2, activation='softmax')(x)

        model = Model(inputs=baseModel.input, outputs=output)
        return model

#with tf.device('/cpu:0'):
model = build_model()
#model = multi_gpu_model(model, 2)
adam =  Adam(0.0001)

def process_data(img_dims, batch_size):
        # Thêm dữ liệu cho data train
        train_datagen = ImageDataGenerator(
        rescale            = 1/255,
        shear_range        = 0.2,
        zoom_range         = 0.2,
        horizontal_flip    = True,
        rotation_range     = 40,
        width_shift_range  = 0.2,
        height_shift_range = 0.2)

        test_datagen       = ImageDataGenerator(rescale=1./255)

        train_generator    = train_datagen.flow_from_directory(
        path_files + 'train',
        target_size        = (img_dims, img_dims),
        batch_size         = batch_size,
        class_mode         = 'categorical')
        
        val_generator      = test_datagen.flow_from_directory(
        path_files + 'val',
        target_size        = (img_dims, img_dims),
        batch_size         = batch_size,
        class_mode         = 'categorical')
        # augumentation cho test
        
        test_generator     = test_datagen.flow_from_directory(
        path_files + 'test',
        target_size        = (img_dims, img_dims),
        batch_size         = batch_size,
        class_mode         = 'categorical')

        return train_generator, val_generator, test_generator

for layer in model.layers:
    layer.trainable = False

tensor_board = TensorBoard(log_dir='Graph/{}'.format(name))

train_generator, val_generator, test_generator = process_data(img_dims, batch_size)
es = EarlyStopping(patience=5)
checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, save_weights_only=True)
model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])
callbacks_list = [es,checkpoint,tensor_board]

H = model.fit_generator(
        train_generator,
        steps_per_epoch    = steps_per_epoch,
        epochs             = epochs,
        validation_data    = val_generator,
        callbacks          = callbacks_list
        )
for layer in baseModel.layers:
    layer.trainable = True
tensor_board = TensorBoard(log_dir='Graph/{}'.format(name))
train_generator, val_generator, test_generator = process_data(img_dims, batch_size)
es = EarlyStopping(patience=5)
checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, save_weights_only=True)
model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])
callbacks_list = [es,checkpoint,tensor_board]
H = model.fit_generator(
        train_generator,
        steps_per_epoch    = steps_per_epoch,
        epochs             = epochs,
        validation_data    = val_generator,
        callbacks          = callbacks_list
        )
test_accuracy = model.evaluate_generator(test_generator, steps=624)