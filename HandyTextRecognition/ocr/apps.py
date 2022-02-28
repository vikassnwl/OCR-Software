from django.apps import AppConfig
from django.conf import  settings
import os


class OcrConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ocr'
    def ready(self):
        os.makedirs(settings.FILE_UPLOAD_PATH, exist_ok=True)
        os.makedirs('ocr/resize/red', exist_ok=True)
        os.makedirs('ocr/resize/morph', exist_ok=True)
        os.makedirs('ocr/outputs', exist_ok=True)
        os.makedirs('ocr/ROI', exist_ok=True)
        
