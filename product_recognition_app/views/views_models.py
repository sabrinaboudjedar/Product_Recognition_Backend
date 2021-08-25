from django.shortcuts import render
from ..functions_model import *
from ..apps import ProductRecognitionAppConfig
from django.http import JsonResponse
# Create your views here.

def run_detector_models(request):
    idUser=request.GET.get('idUser', '')
    path ="imagePro_User"+str(idUser)+".jpg"
    detector= ProductRecognitionAppConfig.detector
    model= ProductRecognitionAppConfig.model_retinaNet
    classes_Faster,scores_Faster,boxes_Faster=fasterRCNN(detector,path)
    reponse=retinaNet(path,idUser,model,classes_Faster,scores_Faster,boxes_Faster)
    return JsonResponse(reponse, safe=False)
