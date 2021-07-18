from django.shortcuts import render
from ..functions_detection_shelf import *
from ..apps import ProductRecognitionAppConfig
from django.http import JsonResponse
# Create your views here.

def run_detector_shelf(request):
    idUser=request.GET.get('idUser', '')
    path ="imagePro_User"+str(idUser)+".jpg"
    nb_shelf=int(request.GET.get('shelf',0))
    detector= ProductRecognitionAppConfig.detector
    model= ProductRecognitionAppConfig.model_retinaNet
    classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille=run_detector_fasterRCNN_first_shelf(detector, path,nb_shelf)
    reponse=run_detector_retinaNet_second_shelf(path,model,idUser,nb_shelf,classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille)
    return JsonResponse(reponse, safe=False)
