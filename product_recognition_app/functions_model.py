from .functions import *
from .shared import *
import base64



def fasterRCNN(detector, path):
    img = load_img(path)
    list_result = {}
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    # start_time = time.time()
    result = detector(converted_img)
    # end_time = time.time()
    i = 0
    classes = []
    boxes = []
    scores = []
    image = Image.open(path)
    size = image.size
    results = {key: value.numpy() for key, value in result.items()}
    classes_result = results["detection_class_entities"]
    scores_result = results["detection_scores"]
    boxes_result = results["detection_boxes"]
    while (i < len(classes_result)):
        if (classes_result[i].decode() in accepted_classes_facter_RCNN):
            shape = [(result["detection_boxes"][i][1] * size[0], result["detection_boxes"][i][0] * size[1]),
                     (result["detection_boxes"][i][3] * size[0], result["detection_boxes"][i][2] * size[1])]
            classes.append(classes_result[i])
            scores.append(1.0)
            boxes.append(boxes_result[i])
            draw = ImageDraw.Draw(image)
            draw.rectangle((shape), fill="#948889")
            image.save('image_facterRCNN_models.jpg', "JPEG")
            image = Image.open('image_facterRCNN_models.jpg')
        i += 1

    return classes, scores, boxes


def retinaNet(path,idUser,inference_model,classes_Faster,scores_Faster,boxes_Faster):
    list_result={}
    classes=[]
    boxes=[]
    scores=[]
    img = load_img(path)
    image_detector=load_img('image_facterRCNN_models.jpg')
    image_pil=Image.open('image_facterRCNN_models.jpg')
    size=image_pil.size
    image = tf.cast(image_detector, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    i=0
    size_im=[]
    while(i<= len(detections.nmsed_classes[0][:num_detections])-1):
            predicted_class=int(detections.nmsed_classes[0][:num_detections][i])
            if((predicted_class in accepted_classes_retinaNet_int )):
                result_boxe=(detections.nmsed_boxes[0][:num_detections] / ratio)[i]
                boxe=result_boxe.numpy()
                boxe=swapPositions(boxe,0,1)
                boxe=swapPositions(boxe,2,3)
                boxe=[boxe[0]/size[1],boxe[1]/size[0],boxe[2]/size[1],boxe[3]/size[0]]
                size_im.append((boxe[3]-boxe[1])*size[0])
                size_im.append((boxe[2]-boxe[0])*size[1])
                if(size_im[0]< 0.4*size[0] and size_im[1]< 0.4*size[1]):
                    classes.append((str(predicted_class)).encode())
                    scores.append(1.0)
                    boxes.append(boxe)
                size_im=[]
            i+=1
    X = np.array(boxes_Faster+boxes)
    list_result["detection_class_entities"]=classes_Faster+classes
    list_result["detection_scores"]=scores_Faster+scores
    list_result["detection_boxes"]= X
    print("le nombre de produits : " ,len(classes))
    image_with_boxes = draw_boxes(
        img.numpy(), list_result["detection_boxes"],
        list_result["detection_class_entities"], list_result["detection_scores"])
    display_image_models(image_with_boxes,idUser)
    with open("image_product_detection_models" + str(idUser) + ".jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    reponse = { "image": image_data}
    return reponse