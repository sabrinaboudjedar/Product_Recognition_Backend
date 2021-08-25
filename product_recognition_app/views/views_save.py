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
from ..functions_classification import *
import json
import pathlib
from .. import config


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

@csrf_exempt
def saveImageCoper(request):
    now = datetime.now()
    data = JSONParser().parse(request)
    classe=data['class']
    idUser=data['idUser']
    if not os.path.exists('./Train'+str(idUser)+'/'+classe):
       os.makedirs('./Train'+str(idUser)+'/'+classe)
    if not os.path.exists('./Test' + str(idUser) + '/' + classe):
        os.makedirs('./Test' + str(idUser) + '/' + classe)
    pathImage=data['pathImCop']
    image_data = re.sub('^data:image/.+;base64,', '', pathImage)
    im = Image.open(BytesIO(base64.b64decode(image_data )))
    path='./Train'+str(idUser)+'/'+classe+'/imageCoper.jpg'
    path_exist='./Train'+str(idUser)+'/'+classe+'/imageCoper'+now.strftime("%m-%d-%Y_%H-%M-%S")+'.jpg'
    path_test = './Test' + str(idUser) + '/' + classe + '/imageCoper.jpg'
    path_exist_test = './Test' + str(idUser) + '/' + classe + '/imageCoper' + now.strftime("%m-%d-%Y_%H-%M-%S") + '.jpg'
    if os.path.exists(path):
        im.save(path_exist, 'JPEG')
        im.save(path_exist_test, 'JPEG')
    else:
        im.save(path, 'JPEG')
        im.save(path_test, 'JPEG')
    reponse="data"
    return JsonResponse(reponse, safe=False)


@csrf_exempt
def returnClassNames(request):
    reponse=class_names
    return JsonResponse(reponse, safe=False)

@csrf_exempt
def get_numberImage(request):
    idUser=request.GET.get('idUser', '')
    data_dir_train,=get_dataSet(idUser)
    image_count=get_countImage(data_dir_train)
    reponse=image_count
    return JsonResponse(reponse, safe=False)

@csrf_exempt
def copyDataset(request):
    data = JSONParser().parse(request)
    idUser=data['idUser']
    if not os.path.exists('./Dataset'+str(idUser)):
        os.makedirs('./Dataset'+str(idUser))
    copy_directory("./Datatset_Products",'./Dataset'+str(idUser))
    reponse="data"
    return JsonResponse(reponse, safe=False)

@csrf_exempt
def save_data(request):
    data = JSONParser().parse(request)
    model_data = data['model']
    idUser = data['idUser']
    model_name=data['name']
    image=data['image']
    if not os.path.exists('./models_rayons'+str(idUser)):
        os.makedirs('./models_rayons'+str(idUser))
    file = pathlib.Path('./models_rayons'+str(idUser)+'/data.json')
    if not file.exists():
        jsonFile = open('./models_rayons' + str(idUser) + '/data.json', "a")
        data = {"models":{}}
        if (model_name not in data["models"]):
            data["models"][model_name]={}
        data["models"][model_name]["data"] = model_data
        data["models"][model_name]["image"]=image
        models = json.dumps(data)
        jsonFile.write(models)
    else:
        jsonFile_reader = open('./models_rayons' + str(idUser) + '/data.json', "r")
        data = json.load(jsonFile_reader)
        if (model_name not in data["models"]):
            data["models"][model_name]={}
        data["models"][model_name]["data"] = model_data
        data["models"][model_name]["image"]=image
        models = json.dumps(data)
        jsonFile_writer = open('./models_rayons' + str(idUser) + '/data.json', "w")
        jsonFile_writer.write(models)



    reponse="data"
    return JsonResponse(reponse, safe=False)


@csrf_exempt
def get_models(request):
    idUser = request.GET.get('idUser', '')
    jsonFile = open('./models_rayons'+str(idUser)+'/data.json',)
    data = json.load(jsonFile)
    print(data)
    if(data!={}):
        my_dict=data.values()
        print(my_dict)
        for dict in my_dict:
            key_list = list(dict.keys())

        reponse=key_list
    else:
        reponse={}
    return JsonResponse(reponse, safe=False)


@csrf_exempt
def get_model_details(request):
    idUser = request.GET.get('idUser', '')
    file = pathlib.Path('./models_rayons' + str(idUser) + '/data.json')
    if not file.exists():
        jsonFile = open('./models_rayons' + str(idUser) + '/data.json', "a")
        data = {"models":{}}
        models = json.dumps(data)
        jsonFile.write(models)
    jsonFile = open('./models_rayons'+str(idUser)+'/data.json',)
    data = json.load(jsonFile)
    reponse=data["models"]
    return JsonResponse(reponse, safe=False)

@csrf_exempt
def get_model(request):
    idUser = request.GET.get('idUser', '')
    model_name = request.GET.get('model_name', '')
    jsonFile = open('./models_rayons'+str(idUser)+'/data.json',)
    data = json.load(jsonFile)
    reponse=data["models"][model_name]
    return JsonResponse(reponse, safe=False)

@csrf_exempt
def delete_model(request):
    idUser = request.GET.get('idUser', '')
    model_name = request.GET.get('model_name', '')
    jsonFile = open('./models_rayons'+str(idUser)+'/data.json',)
    data = json.load(jsonFile)
    del data["models"][model_name]
    jsonFile_writer = open('./models_rayons' + str(idUser) + '/data.json', "w")
    data= json.dumps(data)
    jsonFile_writer.write(data)
    reponse=data
    return JsonResponse(reponse, safe=False)

@csrf_exempt
def post_choice(request):
    data = JSONParser().parse(request)
    selected_choice = data['choice']
    print(selected_choice)
    print(config.choice)
    config.choice=selected_choice
    print(config.choice)
    reponse=True
    return JsonResponse(reponse, safe=False)

