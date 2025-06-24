from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    # Главная страница и авторизация
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    
    # Клиенты
    path('customers/', views.CustomerListView.as_view(), name='customer_list'),
    path('customers/<int:pk>/', views.CustomerDetailView.as_view(), name='customer_detail'),
    path('customers/create/', views.CustomerCreateView.as_view(), name='customer_create'),
    path('customers/<int:pk>/update/', views.CustomerUpdateView.as_view(), name='customer_update'),
    path('customers/<int:pk>/delete/', views.CustomerDeleteView.as_view(), name='customer_delete'),
    
    # Продукты
    path('products/', views.ProductListView.as_view(), name='product_list'),
    path('products/<int:pk>/', views.ProductDetailView.as_view(), name='product_detail'),
    path('products/create/', views.ProductCreateView.as_view(), name='product_create'),
    path('products/<int:pk>/update/', views.ProductUpdateView.as_view(), name='product_update'),
    path('products/<int:pk>/delete/', views.ProductDeleteView.as_view(), name='product_delete'),
    
    # Заказы
    path('orders/', views.OrderListView.as_view(), name='order_list'),
    path('orders/<int:pk>/', views.OrderDetailView.as_view(), name='order_detail'),
    path('orders/create/', views.OrderCreateView.as_view(), name='order_create'),
    path('orders/<int:pk>/update/', views.OrderUpdateView.as_view(), name='order_update'),
    path('orders/<int:pk>/delete/', views.OrderDeleteView.as_view(), name='order_delete'),
    
    # Категории
    path('categories/', views.CategoryListView.as_view(), name='category_list'),
    path('categories/create/', views.CategoryCreateView.as_view(), name='category_create'),
    path('categories/<int:pk>/update/', views.CategoryUpdateView.as_view(), name='category_update'),
    path('categories/<int:pk>/delete/', views.CategoryDeleteView.as_view(), name='category_delete'),
    
    # Регионы
    path('regions/', views.RegionListView.as_view(), name='region_list'),
    path('regions/create/', views.RegionCreateView.as_view(), name='region_create'),
    path('regions/<int:pk>/update/', views.RegionUpdateView.as_view(), name='region_update'),
    path('regions/<int:pk>/delete/', views.RegionDeleteView.as_view(), name='region_delete'),
    
    # Загрузка данных
    path('upload/', views.data_upload, name='data_upload'),
    path('uploads/', views.DataUploadListView.as_view(), name='data_upload_list'),
    path('uploads/<int:upload_id>/process/', views.process_upload, name='process_upload'),
    path('uploads/<int:upload_id>/delete/', views.delete_upload, name='delete_upload'),
    
    # Отчеты
    path('reports/', views.reports_view, name='reports'),
    path('export-report/', views.export_report, name='export_report'),
    path('generate-report/', views.generate_report, name='generate_report'),
    path('reports/list/', views.ReportListView.as_view(), name='report_list'),
    path('reports/<int:pk>/', views.ReportDetailView.as_view(), name='report_detail'),
    path('reports/<int:pk>/delete/', views.ReportDeleteView.as_view(), name='report_delete'),
    
    # API для графиков
    path('api/sales-data/', views.sales_data_api, name='sales_data_api'),
    path('api/product-sales/', views.product_sales_api, name='product_sales_api'),
    path('api/customer-stats/', views.customer_stats_api, name='customer_stats_api'),
] 