from .functions import *
from .shared import *
import base64
import json



def run_detector_zone_models(path,idUser,nb_zone,classes,scores,boxes,result_taille,product_dict,heights):
    img = load_img(path)
    product_dict = json.loads(product_dict)
    result_taille = json.loads(result_taille)
    heights = json.loads(heights)
    classes = classes.replace("'", "\"")
    classes = json.loads(classes)
    result_scores = json.loads(scores)
    result_boxes = json.loads(boxes)
    products = [c.encode() for c in classes]
    zones=[]
    nb=1
    x=[0,1/3,2/3]
    for p in x:
        for z in x:
            zone=[p,z,p+1/3,z+1/3]
            zones.append(zone)
            nb+=1
    remove_classes_zone(products,product_dict, result_boxes,result_scores ,result_taille,zones,nb_zone)
    products.append(("Zone "+str(nb_zone)).encode())
    result_scores.append(1.0)
    boxe=zones[nb_zone-1]
    result_boxes.append(boxe)
    X = np.array(result_boxes)

    nb_produits=len(products)-len(heights)
    image_with_boxes = draw_boxes(
        img.numpy(), X,
        products, result_scores)
    display_image_zone(image_with_boxes,idUser)
    with open("image_product_detection_zone"+str(idUser)+".jpg", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    reponse = {"nb_produits": nb_produits, "produits": product_dict, "etagers": len(heights) - 1,"image":image_data}
    return reponse