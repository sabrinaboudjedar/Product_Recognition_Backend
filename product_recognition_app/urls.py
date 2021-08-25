from  django.urls import path,include
from .views.views import run_detector
from .views.views_save import *
from .views.views_classe import run_detector_classe
from .views.views_shelf import run_detector_shelf
from .views.views_zone import run_detector_zone
from .views.views_planogramme import planogramme
from .views.views_classification import get_classificationModel
from .views.views_product import *
from .views.views_category import *
from .views.views_models import run_detector_models

urlpatterns = [
    path('productRecognition/', run_detector),
    path('productRecognitionSave/', saveImage),
    path('productRecognitionClasse/', run_detector_classe),
    path('productRecognitionShelf/', run_detector_shelf),
    path('productRecognitionZone/', run_detector_zone),
    path('planogramme/', planogramme),
    path('productRecognitionSaveCoper/', saveImageCoper),
    path('productRecognitionClassNames/',returnClassNames),
    path('productRecognitionCountImage/',get_numberImage),
    path('productRecognitionCopyDataset/',copyDataset),
    path('productRecognitionClassification/',get_classificationModel),
    path('productRecognitionSaveFile/', save_data),
    path('productRecognitionGetModels/', get_models),
    path('productRecognitionGetModel/', get_model),
    path('productRecognitionGetModelDetails/', get_model_details),
    path('productRecognitionDeleteModel/', delete_model),
    path('products/', products),
    path('product_details/<int:pk>/', product_details),
    path('categories/', categories),
    path('category_details/<int:pk>/', category_details),
    path('choice/', post_choice),
    path('productRecognitionModels/', run_detector_models),

]