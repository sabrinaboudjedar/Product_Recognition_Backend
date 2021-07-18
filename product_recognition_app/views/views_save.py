from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from PIL import Image
import base64
from base64 import decodestring
from io import BytesIO
import re
from ..shared import *
import os 
from datetime import datetime
from ..functions import *

@csrf_exempt
def saveImage(request):
    data = JSONParser().parse(request)
    pathImage=data['pathIm']
    idUser=data['idUser']
    image_data = re.sub('^data:image/.+;base64,', '', pathImage)
    im = Image.open(BytesIO(base64.b64decode(image_data )))
    im.save('imagePro_User'+str(idUser)+'.jpg', 'JPEG')
    reponse="data"
    return JsonResponse(reponse, safe=False)