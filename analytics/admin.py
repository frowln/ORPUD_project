from django.contrib import admin
from .models import (
    Category, Region, Customer, Product, Order, 
    OrderItem, MLModel, DataUpload
)


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at']
    search_fields = ['name']
    ordering = ['name']


@admin.register(Region)
class RegionAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'population', 'created_at']
    search_fields = ['name', 'code']
    ordering = ['name']


@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ['first_name', 'last_name', 'email', 'region', 'registration_date', 'is_active']
    list_filter = ['region', 'gender', 'is_active', 'registration_date']
    search_fields = ['first_name', 'last_name', 'email']
    ordering = ['-registration_date']
    readonly_fields = ['registration_date']


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'price', 'cost', 'stock_quantity', 'is_active']
    list_filter = ['category', 'is_active', 'created_at']
    search_fields = ['name', 'sku']
    ordering = ['name']
    readonly_fields = ['created_at', 'updated_at']


class OrderItemInline(admin.TabularInline):
    model = OrderItem
    extra = 1
    readonly_fields = ['total_price']


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['order_number', 'customer', 'status', 'total_amount', 'order_date']
    list_filter = ['status', 'order_date']
    search_fields = ['order_number', 'customer__first_name', 'customer__last_name']
    ordering = ['-order_date']
    readonly_fields = ['order_number', 'order_date']
    inlines = [OrderItemInline]


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'algorithm', 'accuracy', 'created_by', 'created_at']
    list_filter = ['model_type', 'is_active', 'created_at']
    search_fields = ['name', 'algorithm']
    ordering = ['-created_at']
    readonly_fields = ['created_at']


@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    list_display = ['name', 'file_type', 'uploaded_by', 'uploaded_at', 'processed', 'row_count']
    list_filter = ['file_type', 'processed', 'uploaded_at']
    search_fields = ['name']
    ordering = ['-uploaded_at']
    readonly_fields = ['uploaded_at']
