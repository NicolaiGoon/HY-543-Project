from matplotlib.cbook import flatten
from torch import classes
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v3 import MobileNetV3Small
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
from keras.layers import Dense, Flatten,GlobalAveragePooling2D
from keras.models import Model,Sequential
import numpy as np


def classificationLayers(model):
    """
    adds classification layers on the top of the model
    """
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def avgPoolingClassifier(model):
  '''
  adds classification layer with avg pooling
  '''
  model.add(GlobalAveragePooling2D())
  model.add(Dense(2, activation='softmax'))
  return model



# -------------- DEFINE MODELS -------------- #
base_resnet50 = ResNet50(
    weights="imagenet", include_top=False, input_shape=(256, 256, 3), classes=2
)

base_MobileNetV3Small = MobileNetV3Small(
    weights="imagenet", include_top=False, input_shape=(256, 256, 3), classes=2
)

base_efficientnetv2s = tf.keras.applications.EfficientNetV2S(
    weights="imagenet", include_top=False, input_shape=(256, 256, 3), classes=2
    )

base_xception = tf.keras.applications.Xception(
  weights="imagenet", include_top=False, input_shape=(256, 256, 3), classes=2
)



# use fine tuning to train further the base_model


fine_tuning = False
classifier = 'avgpool'

assert(classifier == 'default' or classifier == 'avgpool')

if(fine_tuning == False):
    for layer in base_resnet50.layers:
        layer.trainable = False
    for layer_Mobile in base_MobileNetV3Small.layers:
        layer_Mobile.trainable = False
    for layer_effic in  base_efficientnetv2s.layers:
        layer_effic.trainable = False
    for layer_xception in base_xception.layers:
        layer_xception.trainable = False    

model = Sequential()
model.add(base_resnet50)
if classifier == 'default':
  model = classificationLayers(model)
elif classifier == 'avgpool':
  model = avgPoolingClassifier(model)

model_MobileNetV3Small = Sequential()
model_MobileNetV3Small.add(base_MobileNetV3Small)
if classifier == 'default':
  model_MobileNetV3Small = classificationLayers(model_MobileNetV3Small)
elif classifier == 'avgpool':
  model_MobileNetV3Small = avgPoolingClassifier(model_MobileNetV3Small)

model_efficientnetv2s = Sequential()
model_efficientnetv2s.add(base_efficientnetv2s)
if classifier == 'default':
  model_efficientnetv2s = classificationLayers(model_efficientnetv2s)
elif classifier == 'avgpool':
  model_efficientnetv2s = avgPoolingClassifier(model_efficientnetv2s)

model_xception = Sequential()
model_xception.add(base_xception)
if classifier == 'default':
  model_xception = classificationLayers(model_xception)
elif classifier == 'avgpool':
  model_xception = avgPoolingClassifier(model_xception)

# -------------- COLLECT DATA -------------- #

dataPath = "./data/chest_xray" 

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataPath + "/train/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=0,
    validation_split=0.2,
    subset="training"
)

val_ds  = tf.keras.utils.image_dataset_from_directory(
    dataPath + "/train/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=0,
    validation_split=0.2,
    subset="validation"
)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     dataPath + "/val/",
#     labels="inferred",
#     label_mode="categorical",
#     batch_size=32,
#     image_size=(256, 256),
# )

test_ds = tf.keras.utils.image_dataset_from_directory(
    dataPath + "/test/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256),
)

# -------------- TRAIN -------------- # 
# 5 epochs are enough for the problem at hand.

epochs = 10

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.AUC()])
model.fit(train_ds, epochs=epochs, batch_size=32,validation_data=val_ds)
results=model.evaluate(test_ds)
print(dict(zip(model.metrics_names, results)))

model_MobileNetV3Small.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.AUC()])
model_MobileNetV3Small.fit(train_ds, epochs=5, batch_size=32,validation_data=val_ds)
results=model_MobileNetV3Small.evaluate(test_ds)
print(dict(zip(model_MobileNetV3Small.metrics_names, results)))

model_efficientnetv2s.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.AUC()])
model_efficientnetv2s.fit(train_ds, epochs=5, batch_size=32,validation_data=val_ds)
results=model_efficientnetv2s.evaluate(test_ds)
print(dict(zip(model_efficientnetv2s.metrics_names, results)))

model_xception.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.AUC()])
model_xception.fit(train_ds, epochs=5, batch_size=32,validation_data=val_ds)
results=model_xception.evaluate(test_ds)
print(dict(zip(model_xception.metrics_names, results)))