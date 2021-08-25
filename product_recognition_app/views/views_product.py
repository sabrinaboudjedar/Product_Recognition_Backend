from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from ..models import *
from ..serializer import *
import os


@csrf_exempt
def products(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        products = Product.objects.all()
        serializer = ProductSerializer(products , many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        category_name=data['category']
        serializer = ProductSerializer(data=data)
        if serializer.is_valid():
            try:
                Category.objects.get(name=category_name)
            except Category.DoesNotExist:
                data_category={"name":category_name}
                serializer_category = CategorySerializer(data=data_category)
                if serializer_category.is_valid():
                    serializer_category.save()
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        else:
            return JsonResponse(serializer.errors, status=400)

@csrf_exempt
def product_details(request,pk):
    try:
        product = Product.objects.get(pk=pk)
        product_name=product.name
    except Product.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'PUT':
        data_pro = JSONParser().parse(request)
        idUser=data_pro['idUser']
        data=data_pro['data']
        name=data['name']
        serializer = ProductSerializer(product, data=data)
        if serializer.is_valid():
            path = './Train' + str(idUser) + '/' +product_name
            new_path='./Train' + str(idUser) + '/' +name
            path_test='./Test' + str(idUser) + '/' +product_name
            new_path_test = './Test' + str(idUser) + '/' + name
            if os.path.exists(path):
                os.rename(path,new_path)
            if os.path.exists(path_test):
                os.rename(path_test,new_path_test)
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'DELETE':
        idUser = request.GET.get('idUser', '')
        print(product_name)
        product.delete()
        path='./Train'+str(idUser)+'/'+product_name
        path_test='./Test'+str(idUser)+'/'+product_name
        if os.path.exists(path):
            files=os.listdir(path)
            for file in files:
                os.remove(path+'/'+file)
            os.rmdir(path)
        if os.path.exists(path_test):
            files = os.listdir(path_test)
            for file in files:
                os.remove(path_test + '/'+file)
            os.rmdir(path_test)
        return HttpResponse(status=204)


