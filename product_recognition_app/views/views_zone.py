from django.shortcuts import render
from ..functions_detection_zone import *
from ..apps import ProductRecognitionAppConfig
from django.http import JsonResponse
# Create your views here.

def run_detector_zone(request):

    nb_zone=int(request.GET.get('zone', 0))
    idUser = request.GET.get('idUser', '')
    path = "imagePro_User" + str(idUser) + ".jpg"
    classes = request.GET.get('classes')
    scores = request.GET.get('scores')
    boxes = request.GET.get('boxes')
    result_taille = request.GET.get('taille')
    product_dict = request.GET.get('products')
    heights = request.GET.get('height')
    reponse=run_detector_zone_models(path,idUser,nb_zone,classes,scores,boxes,result_taille,product_dict,heights)
    return JsonResponse(reponse, safe=False)
