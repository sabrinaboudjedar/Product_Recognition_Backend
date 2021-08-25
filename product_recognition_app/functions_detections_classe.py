from .functions import *
from .shared import *
import base64
from base64 import decodestring
import json


def run_detector_classe_models(path,idUser,classe_name,classes,scores,boxes,result_taille,product_dict,heights):
    img = load_img(path)

    product_dict=json.loads( product_dict)
    result_taille=json.loads(result_taille)
    heights=json.loads(heights)
    print(classes,type(classes))
    classes=classes.replace("'","\"")
    classes=json.loads(classes)
    print(classes)
    products=[c.encode() for c in classes]
    result_scores=json.loads(scores)
    result_boxes=json.loads(boxes)
    remove_classes_filter(product_dict,classe_name,products,result_boxes,result_scores,result_taille)
    X = np.array(result_boxes)
    nb_produits=len(products)-len(heights)+1
    print(product_dict)

    image_with_boxes = draw_boxes(
        img.numpy(), X,
        products, result_scores)
   
    display_image_classe(image_with_boxes,idUser)
    with open("image_product_detection_classe"+str(idUser)+".jpg", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    reponse = {"nb_produits": nb_produits, "produits": product_dict, "etagers": len(heights) - 1,"image":image_data}
    print(reponse)
    return reponse