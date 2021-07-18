from django.shortcuts import render
from ..functions_detection import *
from ..apps import ProductRecognitionAppConfig
from django.http import JsonResponse
# Create your views here.

def run_detector(request):
    idUser=request.GET.get('idUser', '')
    path ="imagePro_User"+str(idUser)+".jpg"
    detector= ProductRecognitionAppConfig.detector
    model= ProductRecognitionAppConfig.model_retinaNet
    classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille=run_detector_fasterRCNN_first(detector,path)
    reponse=run_detector_retinaNet_second(model,idUser,path,classes_Faster,scores_Faster,boxes_Faster,heights,product_dict,tailles,result_taille)
    return JsonResponse(reponse, safe=False)
