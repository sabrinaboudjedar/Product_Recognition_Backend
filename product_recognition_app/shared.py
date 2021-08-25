from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from . import config

import pathlib

img_height = 150
img_width = 150
batch_size = 16




def get_model_classification():
    print(config.choice)
    if (config.choice):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(THIS_FOLDER, 'model_classification_test.h5')
    else:
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(THIS_FOLDER, 'classification_model_pro.h5')
    model_classification = load_model(MODEL_PATH)

    return model_classification


def get_class_names_products():
    print(config.choice)
    if (config.choice):
        fullPath = os.path.abspath("./" + 'Train')
        data_dir = pathlib.Path(fullPath)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        class_names = train_ds.class_names
    else:
        class_names=['bref', 'bref-3-en-1', 'bref-linge', 'bref-linge-oxy', 'bref-nettoyant', 'bref-power-active','chat-linge-noir-fonce',
             'chat-poudre-haute-performnace', 'chat-poudre-machine', 'chat-poudre-main', 'chat-poudre-universel',
             'chat-power-gel', 'chat-power-gel-main', 'chat-premium-gel', 'chat-savon-de-marseille', 'isis-gel-machine',
             'isis-gel-main', 'isis-poudre-machine', 'isis-poudre-main', 'pril-isis-5-en-1', 'pril-isis-gold', 
             'pril-isis-javel-power', 'pril-isis-peaux-sensibles']
    return class_names



accepted_classes_retinaNet_int=[39,75,73]
accepted_classes_facter_RCNN=['Bottle','Snack']




