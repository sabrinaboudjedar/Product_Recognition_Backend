from django.shortcuts import render
from ..functions import *
from ..apps import ProductRecognitionAppConfig
from django.http import JsonResponse
from ..functions import *
from ..shared import *
import base64
from base64 import decodestring
import json

def planogramme(request):
   
    idUser=request.GET.get('idUser', '')
    shelfs=request.GET.get('dataSrc')
    shelfs_rep=request.GET.get('dataDst')
    shelfs=json.loads(shelfs)
    shelfs_rep=json.loads(shelfs_rep)
    img = load_img('imagePro_User'+str(idUser)+'.jpg')
    classes=[]
    scores=[]
    boxes=[]
    shelf_scores=[]
    nb_produits=0
    nb_pro=0
    nb_produits_shelf=0
    nb_pro_shelf=0
    cpt=0
    total_empty=0
    nb_empty=0
    empty=0
    nb_zone=1
    empty_zone={}
    nb_col=1
    shelf_empty={}
    zones=[]
    x=[0,1/3,2/3]
    for p in x:
        for z in x:
            zone=[p,z,p+1/3,z+1/3]
            zones.append(zone)
    for zone in zones:        
        for key,value in shelfs.items():
            for cle,val in value.items():
                if(key in shelfs_rep.keys() and cle in shelfs_rep[key].keys()):
                    nb_empty=len(shelfs_rep[key].items())-len(value.items())
                    if(shelfs[key][cle][0]!=shelfs_rep[key][cle][0]):
                            classes.append("Mauvaise Position".encode())
                            classe="Mauvaise Position"
                            result_boxe=shelfs[key][cle][1]
                            boxe=[float(result_boxe[0]),float(result_boxe[1]),float(result_boxe[2]),float(result_boxe[3])]
                            add_zone_empty_comp(empty_zone,zone,classe,boxe,nb_zone)
                            scores.append(1.0)
                            boxes.append(boxe)
                            nb_produits+=1
                            nb_produits_shelf+=1
                    else: 
                            classes.append("Bonne Position".encode())
                            classe="Bonne Position"
                            result_boxe=shelfs[key][cle][1]
                            boxe=[float(result_boxe[0]),float(result_boxe[1]),float(result_boxe[2]),float(result_boxe[3])]
                            add_zone_empty_comp(empty_zone,zone,classe,boxe,nb_zone)
                            scores.append(1.0)
                            boxes.append(boxe)
                            nb_produits+=1
                            nb_pro+=1
                            nb_produits_shelf+=1
                            nb_pro_shelf+=1

            while(nb_empty>0):
                nb_produits+=1
                nb_empty-=1   

            if(nb_produits_shelf!=0):
                shelf_scores.append(nb_pro_shelf/nb_produits_shelf)
            else:
                shelf_scores.append(0.0)
            nb_produits_shelf=0
            nb_pro_shelf=0
            cpt+=1
        nb_zone+=1
    
    
    print(empty_zone)
    nb_zone=0
    for key,value in empty_zone.items():
        classes.append(key.encode())
        boxes.append(zones[nb_zone])
        if("Bonne Position" in value.keys()):
            
            if("Mauvaise Position" in value.keys()):
                scores.append(value["Bonne Position"]/(value["Bonne Position"]+value["Mauvaise Position"]))
            else:
                scores.append(1.0)
              
               
        else:
                scores.append(0.0)
          
            
                
        nb_zone+=1
    if(nb_produits!=0):
        pourcentage=(nb_pro/nb_produits)*100
    else:
        pourcentage=0
    #add_shelf_comp(heights,classes,scores,boxes,1.0,shelf_scores)
    X = np.array(boxes)
    image_with_boxes = draw_boxes(
    img.numpy(),X,
    classes, scores)
    display_image_planogramme(image_with_boxes,idUser)
    with open("image_product_detection_planogramme"+str(idUser)+".jpg", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    reponse = {"pourcentage":pourcentage,"image":image_data}
    return JsonResponse(reponse, safe=False)