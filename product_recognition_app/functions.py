# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
import PIL
# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
from PIL import ImageOps
# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
# For measuring the inference time.
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from colorthief import ColorThief
from colour import Color
import webcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from math import sqrt
from .shared import *
from distutils.dir_util import copy_tree
from os import listdir
from os.path import isdir


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def hex2name(c):
    h_color = '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2]))
    try:
        nm = webcolors.hex_to_name(h_color, spec='css3')
    except ValueError as v_error:
        rms_lst = []
        for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
            cur_clr = webcolors.hex_to_rgb(img_hex)
            rmse = np.sqrt(mean_squared_error(c, cur_clr))
            rms_lst.append(rmse)

        closest_color = rms_lst.index(min(rms_lst))

        nm = list(webcolors.CSS3_NAMES_TO_HEX.items())[closest_color][0]
    return nm


def get_image(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_colors(path):
    color_thief = ColorThief(path)
    # get the dominant color
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=2)
    return palette[0]


def compare_colors(c1, c2):
    color1_rgb = sRGBColor(c1[0], c1[1], c1[2])
    color2_rgb = sRGBColor(c2[0], c2[1], c2[2])
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    distance = delta_e / 100
    return distance


def similar_colors(c1, c2):
    r1, g1, b1 = c1[0], c1[1], c1[2]
    r2, g2, b2 = c2[0], c2[1], c2[2]
    d = sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)
    p = d / sqrt((255) ** 2 + (255) ** 2 + (255) ** 2)
    return p


def display_image(image,idUser):
    fig = plt.figure(figsize=(20, 20))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('image_product_detection'+str(idUser)+'.jpg')

def display_image_height(image,idUser):
    fig = plt.figure(figsize=(20, 20))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('image_product_detection_height'+str(idUser)+'.jpg')

def display_image_classe(image,idUser):
    fig = plt.figure(figsize=(20, 20))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('image_product_detection_classe'+str(idUser)+'.jpg')

def display_image_shelf(image,idUser):
    fig = plt.figure(figsize=(20, 20))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('image_product_detection_shelf'+str(idUser)+'.jpg')

def display_image_zone(image,idUser):
    fig = plt.figure(figsize=(20, 20))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('image_product_detection_zone'+str(idUser)+'.jpg')

def display_image_comp(image,idUser):
    fig = plt.figure(figsize=(20, 20))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('image_product_detection_comp'+str(idUser)+'.jpg')

def display_image_planogramme(image,idUser):
    fig = plt.figure(figsize=(20, 20))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('image_product_detection_planogramme'+str(idUser)+'.jpg')

def display_image_models(image,idUser):
    fig = plt.figure(figsize=(20, 20))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('image_product_detection_models'+str(idUser)+'.jpg')

def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename

def draw_bounding_box_on_image(image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color,
                                   font,
                                   thickness=8,
                                   display_str_list=()):

        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = top + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width, text_bottom)],
                           fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill="black",
                      font=font)
            text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=550000):
    colors = list(ImageColor.colormap.values())
    # colors =['#00FFFF'',#FFA500','#FF0000', '#1E90FF', '#FFFF00', '#32CD32']

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  25)
    except IOError:
        font = ImageFont.load_default()

        for i in range(min(boxes.shape[0], max_boxes)):
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def predict_img(path, model_classification, name_classes):
    img_ = tf.keras.preprocessing.image.load_img(
        path, target_size=(150, 150)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img_)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model_classification.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    score = np.max(scores)
    index = np.where(scores.numpy() == np.max(scores))[0][0]
    classe = name_classes[index]
    return classe, score


def return_shelfs(heights, y):
    if len(heights) == 0:
        heights.append(y)
    found = False
    for height in heights:
        val = abs(y - height)
        if (val < 0.1):
            found = True
    if (y not in heights and not found):
        heights.append(y)
    return heights


def add_shelf(heights, classes, result_scores, result_boxes, max_height):
    k = 0
    heights.append(max_height)
    heights = sorted(heights)
    while (k <= (len(heights) - 2)):
        classes.append(("Etagere " + str(k + 1)).encode())
        result_scores.append(1.0)
        boxe = [heights[k], 0.01, heights[k] + (heights[k + 1] - heights[k]) * 0.2, 0.99]
        result_boxes.append(boxe)
        k = k + 1


def add_size(result_taille, tailles, classes):
    j = 0
    while (j <= (len(result_taille) - 1)):
        if (result_taille[j] > (tailles[classes[j].decode()] * 0.73)):
            classes[j] = (classes[j].decode() + " grand").encode()
        else:
            classes[j] = (classes[j].decode() + " petit").encode()
        j = j + 1


def add_length(length, classes, result_scores, result_boxes):
    box = [1 - length, 0, (1 - length) + 0.001, 1]
    classes.append(("Longeur " + str(length)).encode())
    result_scores.append(1.0)
    result_boxes.append(box)


def remove_classes(classes, a, result_boxes, result_scores, result_taille, heights, nb_shelf):
    l = 0
    while (l < len(classes)):
        if (abs(result_boxes[l][2] - heights[nb_shelf - 1]) < 0.1):
            l = l + 1
        else:
            a[classes[l].decode()] -= 1
            del classes[l]
            del result_boxes[l]
            del result_scores[l]



def remove_classes_filter(a, classe_name, classes, result_boxes, result_scores, result_taille):
    l = 0
    while (l < len(classes)):
        if (classe_name == classes[l].decode() or 'Etagere' in classes[l].decode()):
            l = l + 1
        else:
            a[classes[l].decode()] -= 1
            del classes[l]
            del result_boxes[l]
            del result_scores[l]



def add_shelf_col(heights, classes, result_boxes, result_taille):
    col = 0
    h = 0
    m = 0
    etagers = {}
    while (h <= (len(heights) - 2)):
        if ("Etagere " + str(h + 1) not in etagers):
            etagers["Etagere " + str(h + 1)] = {}
        while (m <= (len(classes) - 1)):
            if (abs(result_boxes[m][2] - heights[h]) < 0.1):
                col += 1
                boxe=[str(result_boxes[m][0]),str(result_boxes[m][1]),str(result_boxes[m][2]),str(result_boxes[m][3])]
                etagers["Etagere "+ str(h+1)]["Colonne "+str(col)]=[classes[m].decode(),boxe]
            m += 1
        h += 1
        m = 0
        col = 0
    return etagers

def copy_directory(src,dest):
    for f in listdir(src):
        os.path.isdir(f)
        if not os.path.exists(dest+"/"+f):
            copy_tree(src+"/"+f,dest+"/"+f)
            

def add_column(etagers):
    shelfs = {}
    for key, value in etagers.items():
        sorted_value = sorted(value.items(), key=lambda tup: float(tup[1][1][3]))
        etagers[key] = dict(sorted_value)
    for key, value in etagers.items():
        col = 1
        shelfs[key] = {}
        for values in value.values():
            shelfs[key]["Colonne " + str(col)] = values
            col += 1
    return shelfs

def copy_directory(src,dest):
    for f in listdir(src):
        os.path.isdir(f)
        if not os.path.exists(dest+"/"+f):
            copy_tree(src+"/"+f,dest+"/"+f)

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)

def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )

def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )

def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

def visualize_detections(
    image, boxes, classes, scores, figsize=(100, 100), linewidth=15, color=[0, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes

def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[1024, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio

def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id

def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = tf.keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = tf.keras.Sequential([tf.keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(tf.keras.layers.ReLU())
    head.add(
        tf.keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def swapPositions(list, pos1, pos2):
     
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def remove_classes_zone(classes,a,result_boxes,result_scores,result_taille,zones,nb_zone):
    l=0
    while(l<len(classes)):
            if(result_boxes[l][2]>zones[nb_zone-1][2] ):
                y=zones[nb_zone-1][2]-result_boxes[l][0] 
            else:
                if(result_boxes[l][0]<zones[nb_zone-1][0]):
                    y=result_boxes[l][2]-zones[nb_zone-1][0]
                else:
                    y=result_boxes[l][2]-result_boxes[l][0]
                
            if(result_boxes[l][3]>zones[nb_zone-1][3]):
                x=zones[nb_zone-1][3]-result_boxes[l][1]
            else:
                if(result_boxes[l][1]<zones[nb_zone-1][1]):
                    x=result_boxes[l][3]-zones[nb_zone-1][1]
                  
                else:
                    x=result_boxes[l][3]-result_boxes[l][1]
            
            if(x>0.5*(result_boxes[l][3]-result_boxes[l][1]) and y>0.5*(result_boxes[l][2]-result_boxes[l][0]) or 'Etagere' in classes[l].decode()):
                l+=1 
            else:
                a[classes[l].decode()]-=1
                del classes[l]
                del result_boxes[l]
                del result_scores[l]


def add_zone_empty_comp(zone,zones,classe,result_boxe,nb_zone):
       
        if("Zone " +str(nb_zone) not in zone):
            zone["Zone " +str(nb_zone)]={}
            
        if(result_boxe[2]>zones[2] ):
                y=zones[2]-result_boxe[0] 
        else:
            if(result_boxe[0]<zones[0]):
                y=result_boxe[2]-zones[0]
            else:
                y=result_boxe[2]-result_boxe[0]
                
        if(result_boxe[3]>zones[3]):
                x=zones[3]-result_boxe[1]
        else:
            if(result_boxe[1]<zones[1]):
                    x=result_boxe[3]-zones[1]
            else:
                    x=result_boxe[3]-result_boxe[1]
                
        if(x>0.5*(result_boxe[3]-result_boxe[1]) and y>0.5*(result_boxe[2]-result_boxe[0])):
            if(classe not in zone["Zone " +str(nb_zone)]):
                    zone["Zone " +str(nb_zone)][classe]=1
            else:
                    zone["Zone " +str(nb_zone)][classe]+=1


def add_shelf_comp(heights,classes,result_scores,result_boxes,max_height,scores_shelf):
    k=0 
    if (max_height not in heights):
        heights=sorted(heights)
    while(k<=(len(heights)-2)):
        classes.append(("Etagere "+str(k+1)).encode())
        result_scores.append(scores_shelf[k])
        boxe=[heights[k],0.01,heights[k]+(heights[k+1]-heights[k])*0.2,0.99]
        result_boxes.append(boxe)
        k=k+1

def return_shelf_min(heights,classes,result_boxes,result_taille):
    col=0
    h=0
    m=0
    etagers={}
    sizes=[]
    while(h<=(len(heights)-1)):
        if("Etagere "+str(h+1) not in etagers):
            etagers["Etagere "+str(h+1)]={}
        while(m<= (len(classes)-1)):
            if(abs(result_boxes[m][2]-heights[h])<0.1):
                sizes.append(result_taille[m])
            m+=1
        etagers["Etagere "+str(h+1)]=min(sizes)
        h+=1
        sizes=[]
        m=0
    return  etagers


def run_detector_retinaNet(path,inference_model,classes_second,scores_second,boxes_second,product_dict):
    list_result = {}

    a = {}
    boxes = []
    classes = []
    scores = []
    img = load_img(path)
    image_pil = Image.open(path)
    size = image_pil.size
    image = tf.cast(img, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    i = 0
    model_classification = get_model_classification()
    class_names = get_class_names_products()
    while (i <= len(detections.nmsed_classes[0][:num_detections]) - 1):
        predicted_class = int(detections.nmsed_classes[0][:num_detections][i])

        if ((predicted_class in accepted_classes_retinaNet_int)):
            result_boxe = (detections.nmsed_boxes[0][:num_detections] / ratio)[i]
            boxe = result_boxe.numpy()
            boxe = swapPositions(boxe, 0, 1)
            boxe = swapPositions(boxe, 2, 3)
            boxe = [boxe[0] / size[1], boxe[1] / size[0], boxe[2] / size[1], boxe[3] / size[0]]
            border = (boxe[1] * size[0], boxe[0] * size[1], size[0] - (boxe[3] * size[0]),
                      size[1] - (boxe[2] * size[1]))  # ,0, left, up, right, bottom
            shape = [(boxe[1] * size[0], boxe[0] * size[1]), (boxe[3] * size[0], boxe[2] * size[1])]
            im = ImageOps.crop(image_pil, border)
            path_im = 'image.jpg'
            im.save(path_im)
            size_im = im.size
            classe, score = predict_img(path_im, model_classification, class_names)
            if (score > 0.995 and size_im[0] <0.19 * size[0] and size_im[1] <0.19 * size[1]):
                print(classe,boxe)
                draw = ImageDraw.Draw(image_pil)
                draw.rectangle((shape), fill="#948889")
                image_pil.save('new_image_second.jpg', "JPEG")
                image_pil = Image.open('new_image_second.jpg')
                classes.append(classe.encode())
                boxes.append(boxe)
                scores.append(score)
                if (classe in product_dict):
                    product_dict[classe] += 1
                else:
                    product_dict[classe] = 1
        i = i + 1


    classes_result= classes+classes_second
    scores_result = scores+scores_second
    boxes_result = boxes+boxes_second
    print("le nombre de produits : ", len(classes))
    print(product_dict)

    return classes_result, scores_result,boxes_result,product_dict


def run_detector_fasterRCNN(detector, path):
    a = {}
    tailles = {}
    i = 0
    img = load_img(path)
    img_detector = load_img(path)
    image = Image.open(path)
    size = image.size
    converted_img = tf.image.convert_image_dtype(img_detector, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    result = {key: value.numpy() for key, value in result.items()}
    list_result = dict()
    classes = []
    font = ImageFont.load_default()
    result_boxes = []
    result_scores = []
    result_taille = []
    longeurs = []
    heights = []
    model_classification = get_model_classification()
    class_names = get_class_names_products()
    while (i <= (len(result["detection_scores"]) - 1)):
        if ((result["detection_class_entities"][i].decode() in accepted_classes_facter_RCNN)):
            border = (result["detection_boxes"][i][1] * size[0], result["detection_boxes"][i][0] * size[1],
                      size[0] - (result["detection_boxes"][i][3] * size[0]),
                      size[1] - (result["detection_boxes"][i][2] * size[1]))  # ,0, left, up, right, bottom
            shape = [(result["detection_boxes"][i][1] * size[0], result["detection_boxes"][i][0] * size[1]),
                     (result["detection_boxes"][i][3] * size[0], result["detection_boxes"][i][2] * size[1])]
            im = ImageOps.crop(image, border)
            path_im = 'image.jpg'
            im.save(path_im)
            size_im=im.size
            classe, score = predict_img(path_im, model_classification, class_names)
            size = image.size
            if (score > 0.99 and size_im[0] < 0.2 * size[0] and size_im[1] < 0.2 * size[1]):
                print(classe, result["detection_boxes"][i])
                heights = return_shelfs(heights, result["detection_boxes"][i][2])
                draw = ImageDraw.Draw(image)
                draw.rectangle((shape), fill="#948889")
                image.save('image_facterRCNN.jpg', "JPEG")
                image = Image.open('image_facterRCNN.jpg')
                taille = result["detection_boxes"][i][3] - result["detection_boxes"][i][1]
                classes.append(classe.encode())
                result_boxes.append(result["detection_boxes"][i])
                result_scores.append(score)
                result_taille.append(taille)
                if (classe in a):
                    a[classe] += 1
                    if (taille > tailles[classe]):
                        tailles[classe] = taille
                else:
                    a[classe] = 1
                    tailles[classe] = taille

        i = i + 1
    X = np.array(result_boxes)
    list_result["detection_class_entities"] = classes
    list_result["detection_scores"] = result_scores
    list_result["detection_boxes"] = X
    return classes, result_scores, result_boxes, heights, a, tailles, result_taille

