import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

dataset_path="dataset_classification"

datagen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data=datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=16,
    subset='training'
)

val_data=datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=16,
    subset='validation'
)

with open("classify_classes.json","w") as f:
    json.dump(train_data.class_indices,f)

base_model=MobileNetV2(input_shape=(224,224,3),
                       include_top=False,
                       weights='imagenet')

base_model.trainable=False

model=models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data,validation_data=val_data,epochs=15)

model.save("fracture_classification_model.h5")
