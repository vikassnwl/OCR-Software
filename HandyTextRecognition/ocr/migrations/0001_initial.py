# Generated by Django 4.0.1 on 2022-01-24 07:54

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ROICoordinates',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_version', models.CharField(blank=True, default='Landscape Monster Repeat', max_length=250)),
                ('category', models.CharField(blank=True, default='Label', max_length=250)),
                ('type', models.CharField(blank=True, default='Handwriting', max_length=250)),
                ('extracted_characters', models.CharField(blank=True, max_length=250)),
                ('color', models.CharField(blank=True, max_length=250)),
                ('origin_x', models.CharField(blank=True, max_length=250)),
                ('origin_y', models.CharField(blank=True, max_length=250)),
                ('terminating_x', models.CharField(blank=True, max_length=250)),
                ('terminating_y', models.CharField(blank=True, max_length=250)),
                ('additional_properties1', models.CharField(blank=True, max_length=250)),
                ('additional_properties2', models.CharField(blank=True, max_length=250)),
                ('confidence', models.FloatField(blank=True)),
            ],
        ),
    ]
