from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Count, Avg
from django.utils import timezone
from datetime import datetime, timedelta
import json
from analytics.models import Order, Product, Customer, Category, DataUpload


@login_required
def dashboard(request):
    """Главный дэшборд"""
    # Получаем статистику
    total_orders = Order.objects.count()
    total_revenue = Order.objects.filter(status='delivered').aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    
    total_customers = Customer.objects.count()
    total_products = Product.objects.count()
    
    # Статистика за последние 30 дней
    thirty_days_ago = timezone.now() - timedelta(days=30)
    recent_orders = Order.objects.filter(order_date__gte=thirty_days_ago).count()
    recent_revenue = Order.objects.filter(
        order_date__gte=thirty_days_ago,
        status='delivered'
    ).aggregate(total=Sum('total_amount'))['total'] or 0
    
    # Топ продуктов
    top_products = Product.objects.annotate(
        total_sales=Sum('orderitem__total_price')
    ).order_by('-total_sales')[:5]
    
    # Статистика по категориям
    category_stats = Category.objects.annotate(
        product_count=Count('product'),
        total_sales=Sum('product__orderitem__total_price')
    ).order_by('-total_sales')[:5]
    
    # Последние заказы
    recent_order_list = Order.objects.select_related('customer').order_by('-order_date')[:10]
    
    # Последние загрузки данных
    recent_uploads = DataUpload.objects.select_related('uploaded_by').order_by('-uploaded_at')[:5]
    
    context = {
        'total_orders': total_orders,
        'total_revenue': total_revenue,
        'total_customers': total_customers,
        'total_products': total_products,
        'recent_orders': recent_orders,
        'recent_revenue': recent_revenue,
        'top_products': top_products,
        'category_stats': category_stats,
        'recent_order_list': recent_order_list,
        'recent_uploads': recent_uploads,
    }
    
    return render(request, 'dashboard/dashboard.html', context)


@login_required
def analytics_view(request):
    """Страница аналитики"""
    # Фильтры
    period = request.GET.get('period', '30')
    category_id = request.GET.get('category')
    
    # Определяем период
    if period == '7':
        days = 7
    elif period == '90':
        days = 90
    else:
        days = 30
    
    start_date = timezone.now() - timedelta(days=days)
    
    # Базовые запросы
    orders = Order.objects.filter(order_date__gte=start_date)
    if category_id:
        orders = orders.filter(items__product__category_id=category_id).distinct()
    
    # Общая статистика
    total_orders = Order.objects.count()
    total_revenue = Order.objects.filter(status='delivered').aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    active_customers = Customer.objects.filter(order__isnull=False).distinct().count()
    avg_order_value = Order.objects.filter(status='delivered').aggregate(
        avg=Avg('total_amount')
    )['avg'] or 0
    
    # Данные для графиков
    daily_sales = {}
    daily_orders = {}
    
    for order in orders.filter(status='delivered'):
        date = order.order_date.date().isoformat()
        if date in daily_sales:
            daily_sales[date] += float(order.total_amount)
            daily_orders[date] += 1
        else:
            daily_sales[date] = float(order.total_amount)
            daily_orders[date] = 1
    
    # Статистика по статусам заказов
    status_stats = orders.values('status').annotate(
        count=Count('id')
    ).order_by('status')
    
    # Топ клиентов
    top_customers = Customer.objects.annotate(
        total_spent=Sum('order__total_amount'),
        total_orders=Count('order')
    ).filter(total_spent__gt=0).order_by('-total_spent')[:10]
    
    # Топ продукты
    top_products = Product.objects.annotate(
        total_sales=Sum('orderitem__total_price'),
        total_revenue=Sum('orderitem__total_price')
    ).filter(total_sales__gt=0).order_by('-total_sales')[:5]
    
    # Данные для графиков
    sales_months = list(daily_sales.keys())
    sales_values = list(daily_sales.values())
    
    # Данные для графика по категориям
    category_stats = Category.objects.annotate(
        total_sales=Sum('product__orderitem__total_price')
    ).filter(total_sales__gt=0).order_by('-total_sales')[:5]
    
    category_labels = [cat.name for cat in category_stats]
    category_values = [float(cat.total_sales or 0) for cat in category_stats]
    
    context = {
        'total_orders': total_orders,
        'total_revenue': total_revenue,
        'active_customers': active_customers,
        'avg_order_value': avg_order_value,
        'daily_sales': daily_sales,
        'daily_orders': daily_orders,
        'status_stats': status_stats,
        'top_customers': top_customers,
        'top_products': top_products,
        'categories': Category.objects.all(),
        'selected_period': period,
        'selected_category': category_id,
        'sales_months': json.dumps(sales_months),
        'sales_values': json.dumps(sales_values),
        'category_labels': json.dumps(category_labels),
        'category_values': json.dumps(category_values),
    }
    
    return render(request, 'dashboard/analytics.html', context)


@login_required
def reports_view(request):
    """Страница отчетов"""
    report_type = request.GET.get('type', 'sales')
    
    # Общая статистика для всех отчетов
    total_orders = Order.objects.count()
    active_customers = Customer.objects.filter(order__isnull=False).distinct().count()
    total_revenue = Order.objects.filter(status='delivered').aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    
    if report_type == 'sales':
        # Отчет по продажам
        orders = Order.objects.filter(status='delivered')
        avg_order_value = orders.aggregate(avg=Avg('total_amount'))['avg'] or 0
        
        # По месяцам
        monthly_sales = {}
        for order in orders:
            month = order.order_date.strftime('%Y-%m')
            if month in monthly_sales:
                monthly_sales[month] += float(order.total_amount)
            else:
                monthly_sales[month] = float(order.total_amount)
        
        context = {
            'report_type': 'sales',
            'total_orders': total_orders,
            'active_customers': active_customers,
            'total_revenue': total_revenue,
            'avg_order_value': avg_order_value,
            'monthly_sales': monthly_sales,
        }
        
    elif report_type == 'products':
        # Отчет по продуктам
        products = Product.objects.annotate(
            total_sales=Sum('orderitem__total_price'),
            total_quantity=Sum('orderitem__quantity')
        ).filter(total_sales__gt=0).order_by('-total_sales')
        
        context = {
            'report_type': 'products',
            'total_orders': total_orders,
            'active_customers': active_customers,
            'total_revenue': total_revenue,
            'products': products,
        }
        
    elif report_type == 'customers':
        # Отчет по клиентам
        customers = Customer.objects.annotate(
            total_orders=Count('order'),
            total_spent=Sum('order__total_amount')
        ).filter(total_orders__gt=0).order_by('-total_spent')
        
        context = {
            'report_type': 'customers',
            'total_orders': total_orders,
            'active_customers': active_customers,
            'total_revenue': total_revenue,
            'customers': customers,
        }
    
    else:
        context = {
            'report_type': 'sales',
            'total_orders': total_orders,
            'active_customers': active_customers,
            'total_revenue': total_revenue,
        }
    
    return render(request, 'dashboard/reports.html', context)
