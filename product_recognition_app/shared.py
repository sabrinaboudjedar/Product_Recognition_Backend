from tensorflow.keras.models import load_model
import os
import tensorflow_datasets as tfds




THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_FOLDER, 'model_classification_product.h5')
class_names=['bref', 'bref-3-en-1', 'bref-linge', 'bref-linge-oxy', 'bref-nettoyant', 'chat-linge-noir-fonce', 
             'chat-poudre-haute-performnace', 'chat-poudre-machine', 'chat-poudre-main', 'chat-poudre-universel', 
             'chat-power-gel', 'chat-power-gel-main', 'chat-premium-gel', 'chat-savon-de-marseille', 'isis-gel-machine',
             'isis-gel-main', 'isis-poudre-machine', 'isis-poudre-main', 'pril-isis-5-en-1', 'pril-isis-gold', 
             'pril-isis-javel-power', 'pril-isis-peaux-sensibles']
accepted_classes_retinaNet_int=[39,75,73]
accepted_classes_facter_RCNN=['Bottle','Snack']
model_classification = load_model(MODEL_PATH)



