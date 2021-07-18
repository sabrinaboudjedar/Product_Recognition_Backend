from django.shortcuts import render
from ..functions_detections_classe import *
from ..apps import ProductRecognitionAppConfig
from django.http import JsonResponse
# Create your views here.

def run_detector_classe(request):
    idUser=request.GET.get('idUser', '')
    path ="imagePro_User"+str(idUser)+".jpg"
    classe_name=request.GET.get('classe', '')
    detector= ProductRecognitionAppConfig.detector
    model= ProductRecognitionAppConfig.model_retinaNet
    classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille=run_detector_fasterRCNN_first_classe(detector,path,classe_name)
    reponse=run_detector_retinaNet_second_classe(path,model,idUser,classe_name,classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille)
    return JsonResponse(reponse, safe=False)
