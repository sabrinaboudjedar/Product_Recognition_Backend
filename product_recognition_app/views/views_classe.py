from django.shortcuts import render
from ..functions_detections_classe import *
from ..apps import ProductRecognitionAppConfig
from django.http import JsonResponse
# Create your views here.

def run_detector_classe(request):
    idUser=request.GET.get('idUser', '')
    path = "imagePro_User" + str(idUser) + ".jpg"
    classe_name=request.GET.get('classe', '')
    classes=request.GET.get('classes')
    scores = request.GET.get('scores')
    boxes = request.GET.get('boxes')
    result_taille=request.GET.get('taille')
    product_dict=request.GET.get('products')
    heights=request.GET.get('height')
    reponse=run_detector_classe_models(path,idUser,classe_name,classes,scores,boxes,result_taille,product_dict,heights)
    return JsonResponse(reponse, safe=False)
