from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.contrib.auth import login
from django.db.models import Q, Sum, Count, Avg
from django.http import JsonResponse, HttpResponse
import json
import csv
from django.utils import timezone
from datetime import datetime
import pandas as pd
import os
from django.conf import settings

from .models import (
    Customer, Product, Order, Category, Region, 
    OrderItem, MLModel, DataUpload, Report
)
from .forms import (
    CustomerForm, ProductForm, OrderForm, CategoryForm, 
    RegionForm, DataUploadForm, MLModelForm, CustomerSignUpForm,
    OrderItemForm
)


# Главная страница
def home(request):
    return render(request, 'analytics/home.html')


# Регистрация
def signup(request):
    if request.method == 'POST':
        form = CustomerSignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Регистрация прошла успешно!')
            return redirect('dashboard:dashboard')
    else:
        form = CustomerSignUpForm()
    return render(request, 'analytics/signup.html', {'form': form})


# CRUD для клиентов
class CustomerListView(LoginRequiredMixin, ListView):
    model = Customer
    template_name = 'analytics/customer_list.html'
    context_object_name = 'customers'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Customer.objects.all()
        search = self.request.GET.get('search')
        region = self.request.GET.get('region')
        
        if search:
            queryset = queryset.filter(
                Q(first_name__icontains=search) |
                Q(last_name__icontains=search) |
                Q(email__icontains=search)
            )
        
        if region:
            queryset = queryset.filter(region_id=region)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['regions'] = Region.objects.all()
        return context


class CustomerDetailView(LoginRequiredMixin, DetailView):
    model = Customer
    template_name = 'analytics/customer_detail.html'
    context_object_name = 'customer'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        customer = self.get_object()
        context['orders'] = Order.objects.filter(customer=customer).order_by('-order_date')
        context['total_spent'] = Order.objects.filter(customer=customer).aggregate(
            total=Sum('total_amount')
        )['total'] or 0
        return context


class CustomerCreateView(LoginRequiredMixin, CreateView):
    model = Customer
    form_class = CustomerForm
    template_name = 'analytics/customer_form.html'
    success_url = reverse_lazy('analytics:customer_list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Клиент успешно создан!')
        return super().form_valid(form)


class CustomerUpdateView(LoginRequiredMixin, UpdateView):
    model = Customer
    form_class = CustomerForm
    template_name = 'analytics/customer_form.html'
    success_url = reverse_lazy('analytics:customer_list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Клиент успешно обновлен!')
        return super().form_valid(form)


class CustomerDeleteView(LoginRequiredMixin, DeleteView):
    model = Customer
    template_name = 'analytics/customer_confirm_delete.html'
    success_url = reverse_lazy('analytics:customer_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Клиент успешно удален!')
        return super().delete(request, *args, **kwargs)


# CRUD для продуктов
class ProductListView(LoginRequiredMixin, ListView):
    model = Product
    template_name = 'analytics/product_list.html'
    context_object_name = 'products'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Product.objects.all()
        search = self.request.GET.get('search')
        category = self.request.GET.get('category')
        
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) |
                Q(sku__icontains=search)
            )
        
        if category:
            queryset = queryset.filter(category_id=category)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        return context


class ProductDetailView(LoginRequiredMixin, DetailView):
    model = Product
    template_name = 'analytics/product_detail.html'
    context_object_name = 'product'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        product = self.get_object()
        
        # Статистика продаж продукта
        order_items = product.orderitem_set.all()
        total_orders = order_items.count()
        total_revenue = order_items.aggregate(total=Sum('total_price'))['total'] or 0
        total_quantity = order_items.aggregate(total=Sum('quantity'))['total'] or 0
        
        # История заказов
        order_history = order_items.select_related('order__customer').order_by('-order__order_date')[:10]
        
        context.update({
            'total_orders': total_orders,
            'total_revenue': total_revenue,
            'total_quantity': total_quantity,
            'order_history': order_history,
        })
        
        return context


class ProductCreateView(LoginRequiredMixin, CreateView):
    model = Product
    form_class = ProductForm
    template_name = 'analytics/product_form.html'
    success_url = reverse_lazy('analytics:product_list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Продукт успешно создан!')
        return super().form_valid(form)


class ProductUpdateView(LoginRequiredMixin, UpdateView):
    model = Product
    form_class = ProductForm
    template_name = 'analytics/product_form.html'
    success_url = reverse_lazy('analytics:product_list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Продукт успешно обновлен!')
        return super().form_valid(form)


class ProductDeleteView(LoginRequiredMixin, DeleteView):
    model = Product
    template_name = 'analytics/product_confirm_delete.html'
    success_url = reverse_lazy('analytics:product_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Продукт успешно удален!')
        return super().delete(request, *args, **kwargs)


# CRUD для заказов
class OrderListView(LoginRequiredMixin, ListView):
    model = Order
    template_name = 'analytics/order_list.html'
    context_object_name = 'orders'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Order.objects.all()
        status = self.request.GET.get('status')
        customer = self.request.GET.get('customer')
        
        if status:
            queryset = queryset.filter(status=status)
        
        if customer:
            queryset = queryset.filter(customer_id=customer)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['customers'] = Customer.objects.all()
        context['status_choices'] = Order.STATUS_CHOICES
        return context


class OrderDetailView(LoginRequiredMixin, DetailView):
    model = Order
    template_name = 'analytics/order_detail.html'
    context_object_name = 'order'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        order = self.get_object()
        
        # Получаем все товары в заказе
        order_items = order.items.all().select_related('product')
        
        context.update({
            'order_items': order_items,
        })
        
        return context


class OrderCreateView(LoginRequiredMixin, CreateView):
    model = Order
    form_class = OrderForm
    template_name = 'analytics/order_form.html'
    success_url = reverse_lazy('analytics:order_list')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['order_items'] = [OrderItemForm(self.request.POST, prefix=f'item_{i}') 
                                    for i in range(int(self.request.POST.get('item_count', 1)))]
        else:
            context['order_items'] = [OrderItemForm(prefix='item_0')]
        context['products'] = Product.objects.filter(is_active=True)
        return context
    
    def form_valid(self, form):
        context = self.get_context_data()
        order_items = context['order_items']
        
        if form.is_valid() and all(item_form.is_valid() for item_form in order_items):
            order = form.save(commit=False)
            order.save()
            
            # Сохраняем элементы заказа
            total_amount = 0
            for item_form in order_items:
                if item_form.cleaned_data.get('product') and item_form.cleaned_data.get('quantity'):
                    item = item_form.save(commit=False)
                    item.order = order
                    item.unit_price = item.product.price
                    # Вычисляем total_price вручную
                    item.total_price = item.unit_price * item.quantity
                    item.save()
                    total_amount += float(item.total_price)
            
            # Обновляем общую сумму заказа
            order.total_amount = total_amount
            order.save()
            
            messages.success(self.request, 'Заказ успешно создан!')
            return redirect(self.success_url)
        else:
            return self.render_to_response(self.get_context_data(form=form))


class OrderUpdateView(LoginRequiredMixin, UpdateView):
    model = Order
    form_class = OrderForm
    template_name = 'analytics/order_form.html'
    success_url = reverse_lazy('analytics:order_list')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['order_items'] = [OrderItemForm(self.request.POST, prefix=f'item_{i}') 
                                    for i in range(int(self.request.POST.get('item_count', 1)))]
        else:
            # Показываем существующие элементы заказа
            existing_items = self.object.items.all()
            if existing_items:
                context['order_items'] = [OrderItemForm(instance=item, prefix=f'item_{i}') 
                                        for i, item in enumerate(existing_items)]
            else:
                context['order_items'] = [OrderItemForm(prefix='item_0')]
        context['products'] = Product.objects.filter(is_active=True)
        return context
    
    def form_valid(self, form):
        context = self.get_context_data()
        order_items = context['order_items']
        
        if form.is_valid() and all(item_form.is_valid() for item_form in order_items):
            order = form.save(commit=False)
            order.save()
            
            # Удаляем существующие элементы заказа
            order.items.all().delete()
            
            # Сохраняем новые элементы заказа
            total_amount = 0
            for item_form in order_items:
                if item_form.cleaned_data.get('product') and item_form.cleaned_data.get('quantity'):
                    item = item_form.save(commit=False)
                    item.order = order
                    item.unit_price = item.product.price
                    # Вычисляем total_price вручную
                    item.total_price = item.unit_price * item.quantity
                    item.save()
                    total_amount += float(item.total_price)
            
            # Обновляем общую сумму заказа
            order.total_amount = total_amount
            order.save()
            
            messages.success(self.request, 'Заказ успешно обновлен!')
            return redirect(self.success_url)
        else:
            return self.render_to_response(self.get_context_data(form=form))


class OrderDeleteView(LoginRequiredMixin, DeleteView):
    model = Order
    template_name = 'analytics/order_confirm_delete.html'
    success_url = reverse_lazy('analytics:order_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Заказ успешно удален!')
        return super().delete(request, *args, **kwargs)


# CRUD для категорий
class CategoryListView(LoginRequiredMixin, ListView):
    model = Category
    template_name = 'analytics/category_list.html'
    context_object_name = 'categories'


class CategoryCreateView(LoginRequiredMixin, CreateView):
    model = Category
    form_class = CategoryForm
    template_name = 'analytics/category_form.html'
    success_url = reverse_lazy('analytics:category_list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Категория успешно создана!')
        return super().form_valid(form)


class CategoryUpdateView(LoginRequiredMixin, UpdateView):
    model = Category
    form_class = CategoryForm
    template_name = 'analytics/category_form.html'
    success_url = reverse_lazy('analytics:category_list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Категория успешно обновлена!')
        return super().form_valid(form)


class CategoryDeleteView(LoginRequiredMixin, DeleteView):
    model = Category
    template_name = 'analytics/category_confirm_delete.html'
    success_url = reverse_lazy('analytics:category_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Категория успешно удалена!')
        return super().delete(request, *args, **kwargs)


# CRUD для регионов
class RegionListView(LoginRequiredMixin, ListView):
    model = Region
    template_name = 'analytics/region_list.html'
    context_object_name = 'regions'


class RegionCreateView(LoginRequiredMixin, CreateView):
    model = Region
    form_class = RegionForm
    template_name = 'analytics/region_form.html'
    success_url = reverse_lazy('analytics:region_list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Регион успешно создан!')
        return super().form_valid(form)


class RegionUpdateView(LoginRequiredMixin, UpdateView):
    model = Region
    form_class = RegionForm
    template_name = 'analytics/region_form.html'
    success_url = reverse_lazy('analytics:region_list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Регион успешно обновлен!')
        return super().form_valid(form)


class RegionDeleteView(LoginRequiredMixin, DeleteView):
    model = Region
    template_name = 'analytics/region_confirm_delete.html'
    success_url = reverse_lazy('analytics:region_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Регион успешно удален!')
        return super().delete(request, *args, **kwargs)


# Загрузка данных
@login_required
def data_upload(request):
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = DataUpload.objects.create(
                name=form.cleaned_data['name'],
                file=form.cleaned_data['file'],
                file_type=form.cleaned_data['file_type'],
                uploaded_by=request.user
            )
            
            # Автоматически обрабатываем файл
            try:
                process_uploaded_file(upload)
                messages.success(request, 'Файл успешно загружен и обработан!')
            except Exception as e:
                messages.warning(request, f'Файл загружен, но возникла ошибка при обработке: {str(e)}')
            
            return redirect('analytics:data_upload_list')
    else:
        form = DataUploadForm()
    
    return render(request, 'analytics/data_upload.html', {'form': form})


def process_uploaded_file(upload):
    """Обработка загруженного файла"""
    file_path = upload.file.path
    
    if upload.file_type == 'csv':
        df = pd.read_csv(file_path)
    elif upload.file_type == 'excel':
        df = pd.read_excel(file_path)
    elif upload.file_type == 'json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Неподдерживаемый тип файла: {upload.file_type}")
    
    # Обновляем количество строк
    upload.row_count = len(df)
    
    # Анализируем структуру данных
    columns = list(df.columns)
    columns_lower = ' '.join(columns).lower()
    
    print(f"Обработка файла: {upload.name}")
    print(f"Колонки: {columns}")
    print(f"Количество строк: {len(df)}")
    
    # Определяем тип данных и создаем соответствующие записи
    # Приоритет: заказы > продукты > клиенты
    if ('order' in columns_lower or 'sale' in columns_lower or 
        'customer_email' in columns_lower or 'product_sku' in columns_lower):
        print("Определен тип: ЗАКАЗЫ")
        process_order_data(df, upload)
    elif ('product' in columns_lower or 'item' in columns_lower or 
          'sku' in columns_lower or 'price' in columns_lower):
        print("Определен тип: ПРОДУКТЫ")
        process_product_data(df, upload)
    elif ('customer' in columns_lower or 'client' in columns_lower or 
          'email' in columns_lower or 'first_name' in columns_lower):
        print("Определен тип: КЛИЕНТЫ")
        process_customer_data(df, upload)
    else:
        print("Определен тип: ОБЩИЙ АНАЛИЗ")
        # Общий анализ данных
        process_general_data(df, upload)
    
    upload.processed = True
    upload.save()


def process_customer_data(df, upload):
    """Обработка данных о клиентах"""
    # Создаем регионы если их нет
    if 'region' in df.columns:
        regions = df['region'].dropna().unique()
        for region_name in regions:
            Region.objects.get_or_create(name=region_name)
    
    # Создаем клиентов
    for _, row in df.iterrows():
        customer_data = {
            'first_name': row.get('first_name', row.get('name', 'Неизвестно')),
            'last_name': row.get('last_name', ''),
            'email': row.get('email', f"customer_{row.name}@example.com"),
            'phone': row.get('phone', ''),
            'gender': row.get('gender', ''),
            'address': row.get('address', ''),
        }
        
        if 'region' in row and row['region']:
            try:
                region = Region.objects.get(name=row['region'])
                customer_data['region'] = region
            except Region.DoesNotExist:
                pass
        
        Customer.objects.get_or_create(
            email=customer_data['email'],
            defaults=customer_data
        )


def process_product_data(df, upload):
    """Обработка данных о продуктах"""
    created_products = 0
    created_categories = 0
    
    for _, row in df.iterrows():
        try:
            # Создаем категорию если её нет
            category_name = row.get('category', 'Общие')
            if category_name:
                category, created = Category.objects.get_or_create(name=category_name)
                if created:
                    created_categories += 1
            else:
                category = None
            
            # Создаем продукт
            product_data = {
                'name': row.get('name', row.get('product_name', 'Неизвестный продукт')),
                'description': row.get('description', ''),
                'price': row.get('price', 0),
                'cost': row.get('cost', 0),
                'stock_quantity': row.get('stock_quantity', row.get('quantity', 0)),
                'sku': row.get('sku', f"SKU_{row.name}"),
                'category': category,
            }
            
            product, created = Product.objects.get_or_create(
                sku=product_data['sku'],
                defaults=product_data
            )
            
            if created:
                created_products += 1
                
        except Exception as e:
            # Логируем ошибки, но продолжаем обработку
            print(f"Ошибка при обработке продукта в строке {row.name}: {str(e)}")
            continue
    
    # Обновляем описание загрузки
    upload.description = f"Создано {created_products} продуктов, {created_categories} новых категорий"
    upload.save()


def process_order_data(df, upload):
    """Обработка данных о заказах"""
    created_orders = 0
    created_customers = 0
    created_products = 0
    
    print(f"Начинаем обработку заказов. Всего строк: {len(df)}")
    
    for index, row in df.iterrows():
        try:
            print(f"Обрабатываем строку {index + 1}: {row.get('customer_email', 'N/A')}")
            
            # Находим или создаем клиента
            customer_email = row.get('customer_email', row.get('email', f"customer_{index}@example.com"))
            customer_name = row.get('customer_name', 'Неизвестный')
            
            # Разделяем имя на имя и фамилию
            name_parts = customer_name.split(' ', 1)
            first_name = name_parts[0] if name_parts else 'Неизвестный'
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            
            customer, created = Customer.objects.get_or_create(
                email=customer_email,
                defaults={
                    'first_name': first_name,
                    'last_name': last_name,
                    'phone': row.get('phone', ''),
                    'address': row.get('shipping_address', row.get('address', '')),
                }
            )
            
            if created:
                created_customers += 1
                print(f"  Создан новый клиент: {customer_email}")
            
            # Создаем заказ
            order_data = {
                'customer': customer,
                'status': row.get('status', 'pending'),
                'total_amount': row.get('total_amount', row.get('amount', 0)),
                'shipping_address': row.get('shipping_address', row.get('address', '')),
                'notes': row.get('notes', ''),
            }
            
            order = Order.objects.create(**order_data)
            created_orders += 1
            print(f"  Создан заказ #{order.id} на сумму {order.total_amount}")
            
            # Добавляем товары в заказ если есть
            if 'product_sku' in row and row['product_sku']:
                try:
                    product = Product.objects.get(sku=row['product_sku'])
                    print(f"  Найден существующий продукт: {product.name}")
                except Product.DoesNotExist:
                    # Если продукт не найден, создаем его
                    product_name = row.get('product_name', f"Продукт {row['product_sku']}")
                    category_name = row.get('category', 'Общие')
                    
                    # Создаем категорию если её нет
                    if category_name:
                        category, _ = Category.objects.get_or_create(name=category_name)
                    else:
                        category = None
                    
                    # Создаем продукт
                    product = Product.objects.create(
                        name=product_name,
                        sku=row['product_sku'],
                        price=row.get('price', 0),
                        cost=row.get('cost', 0),
                        category=category,
                        stock_quantity=row.get('stock_quantity', 0),
                    )
                    created_products += 1
                    print(f"  Создан новый продукт: {product.name} (SKU: {product.sku})")
                
                # Создаем элемент заказа
                OrderItem.objects.create(
                    order=order,
                    product=product,
                    quantity=row.get('quantity', 1),
                    unit_price=row.get('unit_price', product.price),
                )
                print(f"  Добавлен товар в заказ: {product.name} x {row.get('quantity', 1)}")
                    
        except Exception as e:
            # Логируем ошибки, но продолжаем обработку
            print(f"Ошибка при обработке строки {index + 1}: {str(e)}")
            continue
    
    print(f"Обработка завершена. Создано: {created_orders} заказов, {created_customers} клиентов, {created_products} продуктов")
    
    # Обновляем описание загрузки
    upload.description = f"Создано {created_orders} заказов, {created_customers} новых клиентов, {created_products} новых продуктов"
    upload.save()


def process_general_data(df, upload):
    """Общий анализ данных"""
    # Сохраняем статистику в описание
    stats = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
    }
    
    upload.description = f"Обработано {len(df)} строк, {len(df.columns)} колонок. Статистика: {stats}"
    upload.save()


@login_required
def process_upload(request, upload_id):
    """Обработка конкретной загрузки"""
    upload = get_object_or_404(DataUpload, id=upload_id)
    
    try:
        process_uploaded_file(upload)
        messages.success(request, 'Файл успешно обработан!')
    except Exception as e:
        messages.error(request, f'Ошибка при обработке файла: {str(e)}')
    
    return redirect('analytics:data_upload_list')


@login_required
def delete_upload(request, upload_id):
    """Удаление загрузки"""
    upload = get_object_or_404(DataUpload, id=upload_id)
    
    # Удаляем файл с диска
    if upload.file and os.path.exists(upload.file.path):
        os.remove(upload.file.path)
    
    upload.delete()
    messages.success(request, 'Загрузка успешно удалена!')
    
    return redirect('analytics:data_upload_list')


class DataUploadListView(LoginRequiredMixin, ListView):
    model = DataUpload
    template_name = 'analytics/data_upload_list.html'
    context_object_name = 'uploads'


# API для получения данных для графиков
@login_required
def sales_data_api(request):
    """API для получения данных о продажах"""
    from datetime import timedelta
    
    # Получаем данные за последние 30 дней
    end_date = timezone.now()
    start_date = end_date - timedelta(days=30)
    
    orders = Order.objects.filter(
        order_date__range=[start_date, end_date],
        status='delivered'
    )
    
    # Группируем по дням
    daily_sales = {}
    for order in orders:
        date = order.order_date.date().isoformat()
        if date in daily_sales:
            daily_sales[date] += float(order.total_amount)
        else:
            daily_sales[date] = float(order.total_amount)
    
    return JsonResponse({
        'labels': list(daily_sales.keys()),
        'data': list(daily_sales.values())
    })


@login_required
def product_sales_api(request):
    """API для получения данных о продажах продуктов"""
    products = Product.objects.annotate(
        total_sales=Sum('orderitem__total_price')
    ).order_by('-total_sales')[:10]
    
    return JsonResponse({
        'labels': [p.name for p in products],
        'data': [float(p.total_sales or 0) for p in products]
    })


@login_required
def customer_stats_api(request):
    """API для статистики клиентов"""
    customers = Customer.objects.all()
    data = {
        'total_customers': customers.count(),
        'active_customers': customers.filter(is_active=True).count(),
        'new_customers_this_month': customers.filter(
            registration_date__month=timezone.now().month
        ).count(),
    }
    return JsonResponse(data)


@login_required
def reports_view(request):
    """Страница отчетов и аналитики"""
    # Получаем данные для отчетов
    total_revenue = Order.objects.filter(status='delivered').aggregate(total=Sum('total_amount'))['total'] or 0
    total_orders = Order.objects.count()
    active_customers = Customer.objects.filter(order__isnull=False).distinct().count()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    # Данные для графиков
    sales_data = Order.objects.filter(status='delivered').values('order_date__month').annotate(
        revenue=Sum('total_amount')
    ).order_by('order_date__month')
    
    sales_months = [item['order_date__month'] for item in sales_data]
    sales_values = [float(item['revenue']) for item in sales_data]
    
    category_data = OrderItem.objects.values('product__category__name').annotate(
        revenue=Sum('total_price')
    ).order_by('-revenue')
    
    category_labels = [item['product__category__name'] for item in category_data]
    category_values = [float(item['revenue']) for item in category_data]
    
    # Топ продуктов и клиентов
    top_products = Product.objects.annotate(
        total_sales=Count('orderitem'),
        total_revenue=Sum('orderitem__total_price')
    ).filter(total_revenue__gt=0).order_by('-total_revenue')[:10]
    
    top_customers = Customer.objects.annotate(
        total_orders=Count('order'),
        total_spent=Sum('order__total_amount')
    ).filter(total_spent__gt=0).order_by('-total_spent')[:10]
    
    context = {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'active_customers': active_customers,
        'avg_order_value': avg_order_value,
        'sales_months': json.dumps(sales_months),
        'sales_values': json.dumps(sales_values),
        'category_labels': json.dumps(category_labels),
        'category_values': json.dumps(category_values),
        'top_products': top_products,
        'top_customers': top_customers,
        'categories': Category.objects.all(),
    }
    
    return render(request, 'analytics/reports.html', context)


@login_required
def export_report(request):
    """Экспорт отчета в Excel/CSV"""
    report_type = request.GET.get('type', 'sales')
    
    if report_type == 'sales':
        # Экспорт данных о продажах
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="sales_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Номер заказа', 'Клиент', 'Дата заказа', 'Статус', 'Сумма', 'Продукты'])
        
        orders = Order.objects.select_related('customer').prefetch_related('items__product').all()
        for order in orders:
            products = ', '.join([f"{item.product.name} (x{item.quantity})" for item in order.items.all()])
            writer.writerow([
                order.order_number,
                order.customer.full_name,
                order.order_date.strftime('%Y-%m-%d %H:%M'),
                order.get_status_display(),
                order.total_amount,
                products
            ])
        
        messages.success(request, 'Отчет о продажах успешно экспортирован!')
        return response
        
    elif report_type == 'products':
        # Экспорт данных о продуктах
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="products_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Название', 'Категория', 'Цена', 'Себестоимость', 'Остаток', 'Продажи', 'Выручка'])
        
        products = Product.objects.select_related('category').annotate(
            total_sales=Count('orderitem'),
            total_revenue=Sum('orderitem__total_price')
        ).all()
        
        for product in products:
            writer.writerow([
                product.name,
                product.category.name,
                product.price,
                product.cost,
                product.stock_quantity,
                product.total_sales or 0,
                product.total_revenue or 0
            ])
        
        messages.success(request, 'Отчет о продуктах успешно экспортирован!')
        return response
    
    else:
        messages.error(request, 'Неизвестный тип отчета!')
        return redirect('analytics:reports')


@login_required
def generate_report(request):
    """Генерация нового отчета"""
    report_type = request.GET.get('type', 'comprehensive')
    
    # Создаем отчет
    if report_type == 'comprehensive':
        # Комплексный отчет
        total_revenue = Order.objects.filter(status='delivered').aggregate(total=Sum('total_amount'))['total'] or 0
        total_orders = Order.objects.count()
        active_customers = Customer.objects.filter(order__isnull=False).distinct().count()
        
        report_content = f"""
КОМПЛЕКСНЫЙ ОТЧЕТ О ПРОДАЖАХ
Дата генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}

ОБЩАЯ СТАТИСТИКА:
- Общая выручка: {total_revenue:,.2f} ₽
- Количество заказов: {total_orders}
- Активных клиентов: {active_customers}

ТОП-5 ПРОДУКТОВ:
"""
        
        top_products = Product.objects.annotate(
            total_sales=Count('orderitem'),
            total_revenue=Sum('orderitem__total_price')
        ).filter(total_revenue__gt=0).order_by('-total_revenue')[:5]
        
        for i, product in enumerate(top_products, 1):
            report_content += f"{i}. {product.name} - {product.total_revenue:,.2f} ₽ ({product.total_sales} продаж)\n"
        
        report_content += "\nТОП-5 КЛИЕНТОВ:\n"
        
        top_customers = Customer.objects.annotate(
            total_orders=Count('order'),
            total_spent=Sum('order__total_amount')
        ).filter(total_spent__gt=0).order_by('-total_spent')[:5]
        
        for i, customer in enumerate(top_customers, 1):
            report_content += f"{i}. {customer.full_name} - {customer.total_spent:,.2f} ₽ ({customer.total_orders} заказов)\n"
        
        report_name = f"Комплексный отчет {datetime.now().strftime('%d.%m.%Y_%H:%M')}"
        
    elif report_type == 'sales':
        # Отчет по продажам
        orders = Order.objects.filter(status='delivered').order_by('-order_date')[:20]
        
        report_content = f"""
ОТЧЕТ ПО ПРОДАЖАМ
Дата генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}

ПОСЛЕДНИЕ 20 ЗАКАЗОВ:
"""
        
        for order in orders:
            report_content += f"- {order.order_number}: {order.customer.full_name} - {order.total_amount:,.2f} ₽ ({order.order_date.strftime('%d.%m.%Y')})\n"
        
        report_name = f"Отчет по продажам {datetime.now().strftime('%d.%m.%Y_%H:%M')}"
    
    else:
        messages.error(request, 'Неизвестный тип отчета!')
        return redirect('analytics:reports')
    
    # Сохраняем отчет в базу данных
    report = Report.objects.create(
        name=report_name,
        content=report_content,
        report_type=report_type,
        generated_by=request.user
    )
    
    messages.success(request, f'Отчет "{report_name}" успешно сгенерирован!')
    return redirect('analytics:report_detail', pk=report.pk)


class ReportListView(LoginRequiredMixin, ListView):
    """Список отчетов"""
    model = Report
    template_name = 'analytics/report_list.html'
    context_object_name = 'reports'
    paginate_by = 20


class ReportDetailView(LoginRequiredMixin, DetailView):
    """Детальная страница отчета"""
    model = Report
    template_name = 'analytics/report_detail.html'
    context_object_name = 'report'


class ReportDeleteView(LoginRequiredMixin, DeleteView):
    """Удаление отчета"""
    model = Report
    template_name = 'analytics/report_confirm_delete.html'
    success_url = reverse_lazy('analytics:report_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Отчет успешно удален!')
        return super().delete(request, *args, **kwargs)
