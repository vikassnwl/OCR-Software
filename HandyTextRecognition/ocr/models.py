from django.db import models

# Create your models here.

class ROICoordinates(models.Model):
    image_version = models.CharField(max_length=250,blank=True,default='Landscape Monster Repeat')
    category = models.CharField(max_length=250,blank=True,default='Label')
    type = models.CharField(max_length=250,blank=True,default='Handwriting')
    extracted_characters = models.CharField(max_length=250,blank=True)
    color = models.CharField(max_length=250,blank=True)
    origin_x = models.CharField(max_length=250,blank=True)
    origin_y = models.CharField(max_length=250,blank=True)
    terminating_x = models.CharField(max_length=250,blank=True)
    terminating_y = models.CharField(max_length=250,blank=True)
    additional_properties1 = models.CharField(max_length=250,blank=True)
    additional_properties2 = models.CharField(max_length=250,blank=True)
    confidence = models.FloatField(blank=True)


    @classmethod
    def create(cls, extracted_characters,color,origin_x,origin_y,terminating_x, terminating_y,confidence):
        roi = cls(extracted_characters=extracted_characters,
                                        color=color,
                                        origin_x=origin_x,
                                        origin_y=origin_y,
                                        terminating_x=terminating_x,
                                        terminating_y=terminating_y,
                                        confidence=confidence)
        roi.save()
        return roi
