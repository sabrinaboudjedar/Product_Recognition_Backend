from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from ..models import *
from ..serializer import *
from django.db.models import Count


@csrf_exempt
def categories(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        categories = Category.objects.all()
        serializer = CategorySerializer(categories , many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = CategorySerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        else:
            print(serializer.errors)
            return JsonResponse(serializer.errors, status=400)


@csrf_exempt
def category_details(request,pk):
    try:
        category = Category.objects.get(pk=pk)
        category_name = category.name
        products = Product.objects.filter(category=category_name)
    except category.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'PUT':
        data = JSONParser().parse(request)
        serializer = CategorySerializer(category, data=data)
        if serializer.is_valid():
            for product in products:
                data_product={"image":product.image,"name":product.name,"category":data['name']}
                serializer_product = ProductSerializer(product,data=data_product)
                if serializer_product.is_valid():
                    serializer_product.save()
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'DELETE':

        for product in products:
            print(product)
            product.delete()
        category.delete()

        return HttpResponse(status=204)


