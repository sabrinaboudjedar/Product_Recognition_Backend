from django.shortcuts import render
from ..functions_detection_zone import *
from ..apps import ProductRecognitionAppConfig
from django.http import JsonResponse
# Create your views here.

def run_detector_zone(request):
    idUser=request.GET.get('idUser', '')
    path ="imagePro_User"+str(idUser)+".jpg"
    zone=int(request.GET.get('zone', 0))
    detector= ProductRecognitionAppConfig.detector
    model= ProductRecognitionAppConfig.model_retinaNet
    classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille=run_detector_fasterRCNN_first_zone(detector,path)
    reponse=run_detector_retinaNet_second_zone(path,model,idUser,zone,classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille)
    return JsonResponse(reponse, safe=False)
