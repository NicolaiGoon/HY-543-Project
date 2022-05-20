from matplotlib.cbook import flatten
from torch import classes
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Model,Sequential
import numpy as np


def classificationLayers(model):
    """
    adds classification layers on the top of the model
    """
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

# -------------- DEFINE MODELS -------------- #
base_resnet50 = ResNet50(
    weights="imagenet", include_top=False, input_shape=(256, 256, 3), classes=2
)

# use fine tuning to train further the base_model
fine_tuning = False
if(fine_tuning == False):
    for layer in base_resnet50.layers:
        layer.trainable = False

model = Sequential()
model.add(base_resnet50)
model = classificationLayers(model)
model.summary()

# -------------- COLLECT DATA -------------- #

dataPath = "./data/chest_xray" 

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataPath + "/train/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataPath + "/val/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
)

# -------------- TRAIN -------------- # 
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
model.fit(train_ds, epochs=5, batch_size=32,validation_data=val_ds)

