from  django.urls import path,include
from .views.views import run_detector
from .views.views_save import *
from .views.views_classe import run_detector_classe
from .views.views_shelf import run_detector_shelf
from .views.views_zone import run_detector_zone
from .views.views_planogramme import planogramme

urlpatterns = [
    path('productRecognition/', run_detector),
    path('productRecognitionSave/', saveImage),
    path('productRecognitionClasse/', run_detector_classe),
    path('productRecognitionShelf/', run_detector_shelf),
    path('productRecognitionZone/', run_detector_zone),
    path('planogramme/', planogramme)
]