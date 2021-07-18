from .functions import *
from .shared import *
import base64
from base64 import decodestring

def run_detector_fasterRCNN_first_shelf(detector, path,nb_shelf):
    a = {}
    tailles={}
    i=0
    img = load_img(path)
    img_detector=load_img(path)
    image=Image.open(path)
    size= image.size
    converted_img  = tf.image.convert_image_dtype(img_detector, tf.float32)[tf.newaxis, ...]
    #start_time = time.time()
    result = detector(converted_img)
    #end_time = time.time()
    result = {key:value.numpy() for key,value in result.items()}
    list_result= dict()
    classes=[]
    font = ImageFont.load_default()
    result_boxes=[]
    result_scores=[]
    result_taille=[]
    longeurs=[]
    heights=[] 
    while(i<= (len(result["detection_scores"])-1)):
        if((result["detection_class_entities"][i].decode() in accepted_classes_facter_RCNN )):
            border =(result["detection_boxes"][i][1]*size[0],result["detection_boxes"][i][0]*size[1],size[0]-(result["detection_boxes"][i][3]*size[0]),size[1]-(result["detection_boxes"][i][2]*size[1]))#,0, left, up, right, bottom
            shape = [(result["detection_boxes"][i][1]*size[0], result["detection_boxes"][i][0]*size[1]), (result["detection_boxes"][i][3]*size[0],result["detection_boxes"][i][2]*size[1])]
            im=ImageOps.crop(image, border)
            path_im='image.jpg'
            im.save(path_im)
            classe,score=predict_img(path_im,model_classification,class_names)
            size= image.size
            if(score>0.996):
                heights=return_shelfs(heights,result["detection_boxes"][i][2])
                draw = ImageDraw.Draw(image)
                draw.rectangle((shape), fill="#948889")
                image.save('image_facterRCNN.jpg', "JPEG")
                image=Image.open('image_facterRCNN.jpg')
                taille=im.size[1]*im.size[0]
                classes.append(classe.encode())
                result_boxes.append(result["detection_boxes"][i])
                result_scores.append(score)
                result_taille.append(taille)
                if(classe in a): 
                    a[classe]+=1
                    if(taille> tailles[classe]):
                        tailles[classe]=taille          
                else:
                    a[classe]=1
                    tailles[classe]=taille
                              
        i=i+1
    #remove_classes(classes,a,result_boxes,result_scores,result_taille,heights,nb_shelf)
    #classes.append(("Etagere "+str(nb_shelf)).encode())
    #result_scores.append(1.0)
    #boxe=[heights[nb_shelf-1],0.01,heights[nb_shelf-1]+(heights[nb_shelf]-heights[nb_shelf-1])*0.2,0.99]
    #result_boxes.append(boxe) 
    #a["Etagere"]=1
    print("Le nombre total des produits est:" , len(classes)-1)
    X = np.array(result_boxes)
    list_result["detection_class_entities"]=classes
    list_result["detection_scores"]=result_scores
    list_result["detection_boxes"]= X
    return classes,result_scores,result_boxes,heights,a,tailles,result_taille

def run_detector_retinaNet_second_shelf(path,model,idUser,nb_shelf,classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille):
   
    list_result={}
    a = {}
    boxes=[]
    classes=[]
    products=[]
    scores=[]
    img = load_img(path)
    image_detector=load_img('image_facterRCNN.jpg')
    image_pil=Image.open('image_facterRCNN.jpg')
    size=image_pil.size
    image = tf.cast(image_detector, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = model.predict(input_image)
    num_detections = detections.valid_detections[0]
    i=0
    while(i<= len(detections.nmsed_classes[0][:num_detections])-1):
            predicted_class=int(detections.nmsed_classes[0][:num_detections][i])
            if((predicted_class in accepted_classes_retinaNet_int )):
                result_boxe=(detections.nmsed_boxes[0][:num_detections] / ratio)[i]
                boxe=result_boxe.numpy()
                boxe=swapPositions(boxe,0,1)
                boxe=swapPositions(boxe,2,3)
                boxe=[boxe[0]/size[1],boxe[1]/size[0],boxe[2]/size[1],boxe[3]/size[0]]
                border =(boxe[1]*size[0],boxe[0]*size[1],size[0]-(boxe[3]*size[0]),size[1]-(boxe[2]*size[1]))#,0, left, up, right, bottom
                shape = [(boxe[1]*size[0], boxe[0]*size[1]), (boxe[3]*size[0],boxe[2]*size[1])]
                im=ImageOps.crop(image_pil, border)
                path_im='image.jpg'
                im.save(path_im)
                size_im=im.size
                taille=size_im[1]*size_im[0]
                classe,score=predict_img(path_im,model_classification,class_names)
                if(score>0.999 and size_im[0]< 0.4*size[0] and size_im[1]< 0.4*size[1] ):
                    heights=return_shelfs(heights,boxe[2])
                    draw = ImageDraw.Draw(image_pil)
                    draw.rectangle((shape), fill="#948889")
                    image_pil.save('new_image.jpg', "JPEG")
                    image_pil=Image.open('new_image.jpg')
                    classes.append(classe.encode())
                    boxes.append(boxe)
                    scores.append(score)
                    result_taille.append(taille)
                    if (classe in product_dict):
                        product_dict[classe] += 1
                        if(taille> tailles[classe]):
                            tailles[classe]=taille      
                    else:
                        product_dict[classe] = 1
                        tailles[classe]=taille
            i=i+1
    heights=sorted(heights)
    products=classes_Faster+classes
    boxes=boxes_Faster+boxes
    scores=scores_Faster+scores
    remove_classes(products,product_dict,boxes,scores,result_taille,heights,nb_shelf)
    products.append(("Etagere "+str(nb_shelf)).encode())
    scores.append(1.0)
    boxe=[heights[nb_shelf-1],0.01,heights[nb_shelf-1]+(heights[nb_shelf]-heights[nb_shelf-1])*0.2,0.99]
    boxes.append(boxe) 
    a["Etagere"]=1
    #add_size(result_taille,tailles,products)
    #add_shelf(heights,products,scores,boxes,1.0)
    #a["Etagere"]=len(heights)-1   
    X = np.array(boxes)
    list_result["detection_class_entities"]=products
    list_result["detection_scores"]=scores
    list_result["detection_boxes"]= X
    nb_produits=len(products)-1
    print("le nombre total des produits est : " ,len(products)-1)
    print(product_dict)
    print("Le nombre d'étagères est : ", 1)
    image_with_boxes = draw_boxes(
        img.numpy(), list_result["detection_boxes"],
        list_result["detection_class_entities"], list_result["detection_scores"])
    display_image_shelf(image_with_boxes,idUser)
    with open("image_product_detection_shelf"+str(idUser)+".jpg", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    reponse = {"nb_produits": nb_produits, "produits": product_dict, "etagers": 1,"image":image_data}
    return reponse