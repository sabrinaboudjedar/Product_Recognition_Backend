from django.apps import AppConfig
import os
import tensorflow_hub as hub
import tensorflow as tf
from  .functions import *
from .classes import *
class ProductRecognitionAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'product_recognition_app'
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(THIS_FOLDER, "faster_rcnn_openimages_v4_inception_resnet_v2_1")
    module_handle = os.path.join(THIS_FOLDER, path)
    #module_handle="https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures['default']
    model_dir = os.path.join(THIS_FOLDER, "retinanet")
    label_encoder = LabelEncoder()

    num_classes = 80
    batch_size = 2

    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )
    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)
   
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]
    weights_dir =os.path.join(THIS_FOLDER, "data")
    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    model.load_weights(latest_checkpoint)
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(confidence_threshold=0)(image, predictions)
    model_retinaNet= tf.keras.Model(inputs=image, outputs=detections)
   