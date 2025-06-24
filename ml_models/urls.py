from django.urls import path
from . import views

app_name = 'ml_models'

urlpatterns = [
    path('', views.ml_dashboard, name='dashboard'),
    path('models/', views.model_list, name='model_list'),
    path('models/create/', views.model_create, name='model_create'),
    path('models/<int:pk>/', views.model_detail, name='model_detail'),
    path('models/<int:pk>/update/', views.model_update, name='model_update'),
    path('models/<int:pk>/delete/', views.model_delete, name='model_delete'),
    path('models/<int:pk>/train/', views.model_train, name='model_train'),
    path('models/<int:pk>/predict/', views.model_predict, name='model_predict'),
    path('train/', views.train_model, name='train_model'),
    path('predict/', views.predict_view, name='predict'),
    path('api/performance/', views.model_performance_api, name='performance_api'),
] 