from django.db import models


class Product(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    image = models.TextField()
    name = models.CharField(default='',unique=True, max_length=100)
    category=models.CharField(default='', max_length=100)
    file_name=models.CharField(default='', max_length=100)
    class Meta:
        ordering = ['created']
        
class Category(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    name = models.CharField(default='', unique=True,max_length=100)
    class Meta:
        ordering = ['created']

