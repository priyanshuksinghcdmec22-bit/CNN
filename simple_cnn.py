import tensorflow as tf
from tensorflow.keras import layers, models

def build_simple_cnn(input_shape=(128,128,3), num_classes=2):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(32,3,activation='relu',padding='same')(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(64,activation='relu')(x)
    x=layers.Dropout(0.3)(x)
    outputs=layers.Dense(num_classes,activation='softmax')(x)
    return models.Model(inputs,outputs)