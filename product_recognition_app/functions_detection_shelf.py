from .functions import *
from .shared import *
import base64
import json





def run_detector_shelf_models(path,idUser,nb_shelf,classes,scores,boxes,result_taille,product_dict,heights):

    img = load_img(path)
    product_dict = json.loads(product_dict)
    result_taille = json.loads(result_taille)
    heights = json.loads(heights)
    classes = classes.replace("'", "\"")
    classes = json.loads(classes)

    result_scores = json.loads(scores)
    result_boxes = json.loads(boxes)
    products = [c.encode() for c in classes]
    l = 0
    while (l < len(products)):
        if('Etagere' in products[l].decode()):
            del products[l]
            del result_boxes[l]
            del result_scores[l]
        else:
            l+=1

    remove_classes(products,product_dict,result_boxes,result_scores ,result_taille,heights,nb_shelf)
    products.append(("Etagere "+str(nb_shelf)).encode())
    result_scores.append(1.0)
    boxe=[heights[nb_shelf-1],0.01,heights[nb_shelf-1]+(heights[nb_shelf]-heights[nb_shelf-1])*0.2,0.99]
    result_boxes.append(boxe)
    X = np.array(result_boxes)
    nb_produits=len(products)-1

    image_with_boxes = draw_boxes(
        img.numpy(), X,
        products, result_scores)

    display_image_shelf(image_with_boxes,idUser)
    with open("image_product_detection_shelf"+str(idUser)+".jpg", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    reponse = {"nb_produits": nb_produits, "produits": product_dict, "etagers": 1,"image":image_data}
    return reponse