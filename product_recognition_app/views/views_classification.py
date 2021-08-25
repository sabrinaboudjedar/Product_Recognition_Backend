from ..functions_classification import *
from django.http import JsonResponse


def get_classificationModel(request):
    idUser=request.GET.get('idUser', '')
    data_dir_train,data_dir_test=get_dataSet(idUser)
    image_count=get_countImage(data_dir_train)
    train_ds=get_trainData(data_dir_train)
    val_ds=get_testData(data_dir_train)
    class_names=get_classNames(train_ds)
    model=get_model(len(class_names))
    compile_model(model)
    accuracy=execute_model(model,train_ds,val_ds)
    save_model(model)
    reponse={"image_count":image_count,"accuracy":accuracy}
    return JsonResponse(reponse, safe=False)