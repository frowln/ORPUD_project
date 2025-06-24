#!/usr/bin/env python
"""
Скрипт для заполнения базы данных тестовыми данными
"""
import os
import sys
import django
from datetime import datetime, timedelta
import random
from decimal import Decimal

# Настройка Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'analytics_platform.settings')
django.setup()

from django.contrib.auth.models import User
from analytics.models import Category, Region, Customer, Product, Order, OrderItem
from django.utils import timezone


def create_test_data():
    """Создание тестовых данных"""
    print("Создание тестовых данных...")
    
    # Создание суперпользователя
    if not User.objects.filter(username='admin').exists():
        User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
        print("Создан суперпользователь: admin/admin123")
    
    # Создание регионов
    regions_data = [
        {'name': 'Москва', 'code': 'MSK', 'population': 12506468},
        {'name': 'Санкт-Петербург', 'code': 'SPB', 'population': 5384342},
        {'name': 'Новосибирск', 'code': 'NVS', 'population': 1625631},
        {'name': 'Екатеринбург', 'code': 'EKB', 'population': 1493749},
        {'name': 'Казань', 'code': 'KZN', 'population': 1257391},
    ]
    
    regions = []
    for region_data in regions_data:
        region, created = Region.objects.get_or_create(
            code=region_data['code'],
            defaults=region_data
        )
        regions.append(region)
        if created:
            print(f"Создан регион: {region.name}")
    
    # Создание категорий
    categories_data = [
        {'name': 'Электроника', 'description': 'Компьютеры, телефоны, планшеты'},
        {'name': 'Одежда', 'description': 'Мужская и женская одежда'},
        {'name': 'Книги', 'description': 'Художественная и техническая литература'},
        {'name': 'Спорт', 'description': 'Спортивные товары и оборудование'},
        {'name': 'Дом и сад', 'description': 'Товары для дома и сада'},
    ]
    
    categories = []
    for category_data in categories_data:
        category, created = Category.objects.get_or_create(
            name=category_data['name'],
            defaults=category_data
        )
        categories.append(category)
        if created:
            print(f"Создана категория: {category.name}")
    
    # Создание клиентов
    customers_data = [
        {'first_name': 'Иван', 'last_name': 'Иванов', 'email': 'ivan@example.com', 'phone': '+7-900-123-45-67'},
        {'first_name': 'Мария', 'last_name': 'Петрова', 'email': 'maria@example.com', 'phone': '+7-900-234-56-78'},
        {'first_name': 'Алексей', 'last_name': 'Сидоров', 'email': 'alex@example.com', 'phone': '+7-900-345-67-89'},
        {'first_name': 'Елена', 'last_name': 'Козлова', 'email': 'elena@example.com', 'phone': '+7-900-456-78-90'},
        {'first_name': 'Дмитрий', 'last_name': 'Новиков', 'email': 'dmitry@example.com', 'phone': '+7-900-567-89-01'},
        {'first_name': 'Анна', 'last_name': 'Морозова', 'email': 'anna@example.com', 'phone': '+7-900-678-90-12'},
        {'first_name': 'Сергей', 'last_name': 'Волков', 'email': 'sergey@example.com', 'phone': '+7-900-789-01-23'},
        {'first_name': 'Ольга', 'last_name': 'Лебедева', 'email': 'olga@example.com', 'phone': '+7-900-890-12-34'},
        {'first_name': 'Павел', 'last_name': 'Соколов', 'email': 'pavel@example.com', 'phone': '+7-900-901-23-45'},
        {'first_name': 'Наталья', 'last_name': 'Зайцева', 'email': 'natalia@example.com', 'phone': '+7-900-012-34-56'},
    ]
    
    customers = []
    for customer_data in customers_data:
        customer, created = Customer.objects.get_or_create(
            email=customer_data['email'],
            defaults={
                **customer_data,
                'region': random.choice(regions),
                'gender': random.choice(['M', 'F']),
                'birth_date': datetime.now().date() - timedelta(days=random.randint(6570, 25550)),  # 18-70 лет
                'address': f"ул. Примерная, д. {random.randint(1, 100)}, кв. {random.randint(1, 100)}"
            }
        )
        customers.append(customer)
        if created:
            print(f"Создан клиент: {customer.full_name}")
    
    # Создание продуктов
    products_data = [
        # Электроника
        {'name': 'iPhone 15 Pro', 'category': categories[0], 'price': 99990, 'cost': 70000, 'sku': 'IPH15PRO'},
        {'name': 'MacBook Air M2', 'category': categories[0], 'price': 129990, 'cost': 90000, 'sku': 'MBAIRM2'},
        {'name': 'Samsung Galaxy S24', 'category': categories[0], 'price': 89990, 'cost': 65000, 'sku': 'SGS24'},
        {'name': 'iPad Air', 'category': categories[0], 'price': 69990, 'cost': 50000, 'sku': 'IPADAIR'},
        {'name': 'AirPods Pro', 'category': categories[0], 'price': 24990, 'cost': 18000, 'sku': 'AIRPODSPRO'},
        
        # Одежда
        {'name': 'Джинсы Levi\'s', 'category': categories[1], 'price': 5990, 'cost': 3000, 'sku': 'LEVIS501'},
        {'name': 'Футболка Nike', 'category': categories[1], 'price': 2990, 'cost': 1500, 'sku': 'NIKE001'},
        {'name': 'Кроссовки Adidas', 'category': categories[1], 'price': 8990, 'cost': 4500, 'sku': 'ADIDAS001'},
        {'name': 'Пальто зимнее', 'category': categories[1], 'price': 15990, 'cost': 8000, 'sku': 'COAT001'},
        {'name': 'Платье вечернее', 'category': categories[1], 'price': 12990, 'cost': 6500, 'sku': 'DRESS001'},
        
        # Книги
        {'name': 'Война и мир', 'category': categories[2], 'price': 990, 'cost': 500, 'sku': 'BOOK001'},
        {'name': 'Мастер и Маргарита', 'category': categories[2], 'price': 790, 'cost': 400, 'sku': 'BOOK002'},
        {'name': 'Python для начинающих', 'category': categories[2], 'price': 1490, 'cost': 800, 'sku': 'BOOK003'},
        {'name': 'Алгоритмы и структуры данных', 'category': categories[2], 'price': 1990, 'cost': 1000, 'sku': 'BOOK004'},
        {'name': 'Искусство программирования', 'category': categories[2], 'price': 2990, 'cost': 1500, 'sku': 'BOOK005'},
        
        # Спорт
        {'name': 'Гантели 5кг', 'category': categories[3], 'price': 1990, 'cost': 1000, 'sku': 'SPORT001'},
        {'name': 'Коврик для йоги', 'category': categories[3], 'price': 990, 'cost': 500, 'sku': 'SPORT002'},
        {'name': 'Велосипед горный', 'category': categories[3], 'price': 29990, 'cost': 20000, 'sku': 'SPORT003'},
        {'name': 'Беговая дорожка', 'category': categories[3], 'price': 49990, 'cost': 35000, 'sku': 'SPORT004'},
        {'name': 'Тренажер для пресса', 'category': categories[3], 'price': 3990, 'cost': 2000, 'sku': 'SPORT005'},
        
        # Дом и сад
        {'name': 'Кофемашина', 'category': categories[4], 'price': 19990, 'cost': 12000, 'sku': 'HOME001'},
        {'name': 'Пылесос робот', 'category': categories[4], 'price': 29990, 'cost': 18000, 'sku': 'HOME002'},
        {'name': 'Набор кастрюль', 'category': categories[4], 'price': 5990, 'cost': 3000, 'sku': 'HOME003'},
        {'name': 'Садовые инструменты', 'category': categories[4], 'price': 3990, 'cost': 2000, 'sku': 'HOME004'},
        {'name': 'Гриль электрический', 'category': categories[4], 'price': 8990, 'cost': 5000, 'sku': 'HOME005'},
    ]
    
    products = []
    for product_data in products_data:
        product, created = Product.objects.get_or_create(
            sku=product_data['sku'],
            defaults={
                **product_data,
                'price': Decimal(product_data['price']),
                'cost': Decimal(product_data['cost']),
                'stock_quantity': random.randint(10, 100),
                'description': f"Описание для {product_data['name']}"
            }
        )
        products.append(product)
        if created:
            print(f"Создан продукт: {product.name}")
    
    # Создание заказов
    order_statuses = ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
    status_weights = [0.1, 0.2, 0.2, 0.4, 0.1]  # Больше доставленных заказов
    
    for i in range(50):  # Создаем 50 заказов
        customer = random.choice(customers)
        order_date = timezone.now() - timedelta(days=random.randint(1, 365))
        status = random.choices(order_statuses, weights=status_weights)[0]
        
        order = Order.objects.create(
            customer=customer,
            status=status,
            order_date=order_date,
            delivery_date=order_date + timedelta(days=random.randint(1, 7)) if status in ['shipped', 'delivered'] else None,
            shipping_address=customer.address,
            notes=f"Заказ #{i+1}"
        )
        
        # Создание элементов заказа
        num_items = random.randint(1, 5)
        order_items = random.sample(products, min(num_items, len(products)))
        
        total_amount = Decimal('0')
        for product in order_items:
            quantity = random.randint(1, 3)
            unit_price = product.price
            total_price = unit_price * quantity
            
            OrderItem.objects.create(
                order=order,
                product=product,
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price
            )
            
            total_amount += total_price
        
        order.total_amount = total_amount
        order.save()
        
        print(f"Создан заказ: {order.order_number} на сумму {total_amount} ₽")
    
    print("\nТестовые данные успешно созданы!")
    print(f"Создано:")
    print(f"- {len(regions)} регионов")
    print(f"- {len(categories)} категорий")
    print(f"- {len(customers)} клиентов")
    print(f"- {len(products)} продуктов")
    print(f"- {Order.objects.count()} заказов")


if __name__ == '__main__':
    create_test_data() 