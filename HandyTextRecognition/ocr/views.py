from rest_framework.parsers import FileUploadParser,MultiPartParser,FormParser,JSONParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import  status
from django.conf import  settings
from .models import ROICoordinates
# Create your views here.
import os,sys,time
import pandas as pd
from .alpha import rgb,run
sys.path.append('ocr/src')
from django.http import HttpResponse, HttpResponseNotFound
from termcolor import colored

class FileUploadView(APIView):
    parser_classes = (MultiPartParser,FormParser)
    def post(self, request, format=None):
        
        # Handling invalid file extension and no file chosen exception
        try:
            file_obj = request.FILES['image']

            if str(file_obj).split('.')[-1] not in ('jpg', 'jpeg', 'JPG', 'png'):
                raise Exception("invalid file type or extension")


            with open(f"{settings.FILE_UPLOAD_PATH}{file_obj}", 'wb+') as destination:
                for chunk in file_obj.chunks():
                    destination.write(chunk)

        except:
            # Handling invalid file extension and no file chosen exception
            return Response({"Success":False,"Message":"Invalid file extension or no file chosen or no field name provided"},status=status.HTTP_400_BAD_REQUEST)

        # base_file=f"{settings.FILE_UPLOAD_PATH}{file_obj}"
        rgb(f"{settings.FILE_UPLOAD_PATH}{file_obj}")
        rdata = run()
        roi= ROICoordinates.objects.all()

        rec_prob = []
        for prob in rdata[2]:
            rec_prob.append(prob[1][1])

        rdata[0].to_csv("a.csv",index=False)
        df = pd.read_csv("a.csv")
        print("rdata :   ",rdata)
        print(colored(rdata, 'red'))
        print(colored(rdata[1], 'green'))
        for i ,j in df.iterrows():
            print(rdata[1][i])
            roi = ROICoordinates.create(extracted_characters=rdata[1][i],
                                        color=df.loc[i,"TextColor"],
                                        origin_x=df.loc[i,"x"],
                                        origin_y=df.loc[i,"y"],
                                        terminating_x=df.loc[i,"x+w"],
                                        terminating_y=df.loc[i,"y+h"],
                                        confidence= rec_prob[i]
                                        )
        

        output_img_path = f'{settings.FILE_DOWNLOAD_PATH}basic_input.jpg'
        with open(output_img_path, "rb") as f:
            return HttpResponse(f.read(), content_type="image/jpeg")

        # return Response({"Success":True,"Message":"File Uploaded"},status=status.HTTP_200_OK)
