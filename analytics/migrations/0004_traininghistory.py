# Generated by Django 5.0.2 on 2025-06-22 23:03

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analytics', '0003_mlmodel_last_trained_alter_mlmodel_algorithm'),
    ]

    operations = [
        migrations.CreateModel(
            name='TrainingHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('training_date', models.DateTimeField(auto_now_add=True, verbose_name='Дата обучения')),
                ('accuracy', models.FloatField(verbose_name='Точность')),
                ('dataset_size', models.IntegerField(verbose_name='Размер датасета')),
                ('training_time', models.FloatField(verbose_name='Время обучения (сек)')),
                ('parameters', models.JSONField(default=dict, verbose_name='Параметры обучения')),
                ('model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='training_history', to='analytics.mlmodel', verbose_name='Модель')),
            ],
            options={
                'verbose_name': 'История обучения',
                'verbose_name_plural': 'История обучения',
                'ordering': ['-training_date'],
            },
        ),
    ]
