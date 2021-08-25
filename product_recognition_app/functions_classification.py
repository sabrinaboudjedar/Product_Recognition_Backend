
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


batch_size = 16
img_height = 150
img_width = 150
epochs=10

def get_dataSet(idUser):
    fullPath_train = os.path.abspath("./" + 'Train'+str(idUser))
    #data_dir =  keras.utils.get_file('Dataset_Test_user01', 'file://'+fullPath)
    data_dir_train = pathlib.Path(fullPath_train)
    fullPath_test = os.path.abspath("./" + 'Test' + str(idUser))
    data_dir_test = pathlib.Path(fullPath_test)
    return data_dir_train,data_dir_test

def get_countImage(data_dir):
    image_count = len(list(data_dir.glob('*/*.JPG')))
    return image_count

def get_trainData(data_dir):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=123,
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return train_ds

def get_testData(data_dir):
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=123,
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return val_ds

def get_classNames(train_ds):
    class_names = train_ds.class_names
    return class_names

def get_model(num_classes):
    model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    # Ajout de la première couche de convolution, suivie d'une couche ReLU
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    # Ajout de la première couche de pooling
    layers.MaxPooling2D(),
    # Ajout de la deuxième couche de convolution, suivie d'une couche ReLU
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    # Ajout de la deuxième couche de pooling
    layers.MaxPooling2D(),
    # Ajout de la troisième couche de convolution, suivie d'une couche ReLU
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    # Ajout de la deuxième couche de pooling
    layers.MaxPooling2D(),
    # Conversion des matrices 3D en vecteur  
    layers.Flatten(),
    # Ajout de la première couche fully-connected, suivie d'une couche ReLU
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])
    return model

def compile_model(model):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.summary()

def execute_model(model,train_ds,val_ds):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
        )
    val_acc = history.history['val_accuracy']
    return val_acc[len(val_acc)-1]

def save_model(model):
    model.save("./product_recognition_app/model_classification_test.h5")



