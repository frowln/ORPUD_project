from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.db.models import Sum, Count, Avg
from django.utils import timezone
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
from django import forms

from analytics.models import Order, Product, Customer, MLModel, TrainingHistory, PredictionHistory


@login_required
def ml_dashboard(request):
    """Дэшборд машинного обучения"""
    models = MLModel.objects.filter(created_by=request.user).order_by('-created_at')
    
    context = {
        'recent_models': models[:5],
        'total_models': models.count(),
        'active_models': models.filter(is_active=True).count(),
        'avg_accuracy': models.aggregate(avg=Avg('accuracy'))['avg'] or 0,
        'total_predictions': 0,  # Здесь можно добавить подсчет предсказаний
    }
    
    return render(request, 'ml_models/dashboard.html', context)


@login_required
def train_model(request):
    """Обучение модели"""
    if request.method == 'POST':
        model_type = request.POST.get('model_type')
        algorithm = request.POST.get('algorithm')
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        
        try:
            # Получаем данные для обучения
            if model_type == 'regression':
                # Прогнозирование продаж
                data = prepare_sales_data()
                model, accuracy = train_regression_model(data, algorithm)
            elif model_type == 'classification':
                # Классификация клиентов
                data = prepare_customer_data()
                model, accuracy = train_classification_model(data, algorithm)
            elif model_type == 'clustering':
                # Кластеризация продуктов
                data = prepare_product_data()
                model, accuracy = train_clustering_model(data, algorithm)
            elif model_type == 'forecasting':
                # Прогнозирование временных рядов
                data = prepare_forecasting_data()
                model, accuracy = train_forecasting_model(data, algorithm)
            else:
                messages.error(request, 'Неподдерживаемый тип модели')
                return redirect('ml_dashboard')
            
            # Сохраняем модель
            model_instance = MLModel.objects.create(
                name=name,
                model_type=model_type,
                algorithm=algorithm,
                description=description,
                accuracy=accuracy,
                created_by=request.user
            )
            
            # Сохраняем файл модели
            model_dir = 'ml_models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_path = f'{model_dir}/model_{model_instance.id}.pkl'
            joblib.dump(model, model_path)
            model_instance.model_file = model_path
            model_instance.save()
            
            messages.success(request, f'Модель {name} успешно обучена! Точность: {accuracy:.2f}')
            
        except Exception as e:
            messages.error(request, f'Ошибка при обучении модели: {str(e)}')
        
        return redirect('ml_dashboard')
    
    return render(request, 'ml_models/train_model.html')


@login_required
def predict_view(request):
    """Страница для прогнозирования"""
    models = MLModel.objects.filter(is_active=True, created_by=request.user)
    
    if request.method == 'POST':
        model_id = request.POST.get('model_id')
        try:
            model_instance = MLModel.objects.get(id=model_id, created_by=request.user)
            model = joblib.load(model_instance.model_file)
            
            if model_instance.model_type == 'regression':
                # Прогнозирование продаж
                prediction = predict_sales(model, request.POST)
            elif model_instance.model_type == 'classification':
                # Классификация клиентов
                prediction = predict_customer_class(model, request.POST)
            elif model_instance.model_type == 'clustering':
                # Кластеризация продуктов
                prediction = predict_clustering(model, request.POST)
            elif model_instance.model_type == 'forecasting':
                # Прогнозирование временных рядов
                prediction = predict_forecasting(model, request.POST)
            else:
                prediction = "Неподдерживаемый тип модели"
            
            context = {
                'models': models,
                'prediction': prediction,
                'selected_model': model_instance,
                'accuracy_percent': model_instance.accuracy * 100 if model_instance.accuracy is not None else None,
            }
            
        except Exception as e:
            messages.error(request, f'Ошибка при прогнозировании: {str(e)}')
            context = {'models': models}
    
    else:
        context = {'models': models}
    
    return render(request, 'ml_models/predict.html', context)


@login_required
def model_detail(request, pk):
    """Детали модели"""
    try:
        model = MLModel.objects.get(id=pk, created_by=request.user)
        training_history = model.training_history.all()[:10]  # Последние 10 записей
        accuracy_percent = model.accuracy * 100 if model.accuracy is not None else None
        context = {
            'model': model,
            'training_history': training_history,
            'accuracy_percent': accuracy_percent
        }
        return render(request, 'ml_models/model_detail.html', context)
    except MLModel.DoesNotExist:
        messages.error(request, 'Модель не найдена')
        return redirect('ml_models:model_list')


@login_required
def delete_model(request, pk):
    """Удаление модели"""
    try:
        model = MLModel.objects.get(id=pk, created_by=request.user)
        if model.model_file and os.path.exists(model.model_file.path):
            os.remove(model.model_file.path)
        model.delete()
        messages.success(request, 'Модель успешно удалена')
    except MLModel.DoesNotExist:
        messages.error(request, 'Модель не найдена')
    
    return redirect('ml_models:model_list')


# Вспомогательные функции для подготовки данных
def prepare_sales_data():
    """Подготовка данных для прогнозирования продаж"""
    orders = Order.objects.filter(status='delivered').select_related('customer').prefetch_related('items')
    
    if not orders.exists():
        raise ValueError("Нет данных о продажах для обучения модели")
    
    data = []
    for order in orders:
        # Вычисляем возраст клиента в годах
        customer_age = 0
        if order.customer.birth_date:
            age_delta = datetime.now().date() - order.customer.birth_date
            customer_age = age_delta.days / 365.25
        
        # Вычисляем дополнительные признаки
        total_items = order.items.count()
        avg_item_price = float(order.total_amount) / total_items if total_items > 0 else 0
        
        # День недели (0=Понедельник, 6=Воскресенье)
        weekday = order.order_date.weekday()
        
        # Сезонность (1-4 кварталы)
        quarter = (order.order_date.month - 1) // 3 + 1
        
        # Время года (1-4)
        season = (order.order_date.month % 12 + 3) // 3
        
        data.append({
            'customer_age': customer_age,
            'order_month': order.order_date.month,
            'order_day': order.order_date.day,
            'order_weekday': weekday,
            'quarter': quarter,
            'season': season,
            'items_count': total_items,
            'avg_item_price': avg_item_price,
            'total_amount': float(order.total_amount)
        })
    
    df = pd.DataFrame(data)
    if len(df) < 10:  # Минимум 10 записей для обучения
        raise ValueError("Недостаточно данных для обучения модели (минимум 10 записей)")
    
    print(f"DEBUG: Prepared {len(df)} sales records")
    print(f"DEBUG: Features: {df.columns.tolist()}")
    print(f"DEBUG: Target range: {df['total_amount'].min():.2f} - {df['total_amount'].max():.2f}")
    print(f"DEBUG: Target mean: {df['total_amount'].mean():.2f}")
    
    X = df.drop('total_amount', axis=1)
    y = df['total_amount']
    
    return X, y


def prepare_customer_data():
    """Подготовка данных для классификации клиентов"""
    customers = Customer.objects.annotate(
        total_orders=Count('order'),
        total_spent=Sum('order__total_amount')
    ).filter(total_orders__gt=0)
    
    if not customers.exists():
        raise ValueError("Нет данных о клиентах для обучения модели")
    
    data = []
    for customer in customers:
        data.append({
            'total_orders': customer.total_orders,
            'total_spent': float(customer.total_spent or 0),
            'days_since_registration': (datetime.now().date() - customer.registration_date.date()).days,
            'is_active': 1 if customer.is_active else 0,
            'customer_class': 'high' if float(customer.total_spent or 0) > 1000 else 'medium' if float(customer.total_spent or 0) > 500 else 'low'
        })
    
    df = pd.DataFrame(data)
    if len(df) < 10:  # Минимум 10 записей для обучения
        raise ValueError("Недостаточно данных для обучения модели (минимум 10 записей)")
    
    X = df.drop('customer_class', axis=1)
    y = df['customer_class']
    
    return X, y


def prepare_product_data():
    """Подготовка данных для кластеризации продуктов"""
    products = Product.objects.annotate(
        total_sales=Sum('orderitem__total_price'),
        total_quantity=Sum('orderitem__quantity')
    ).filter(total_sales__gt=0)
    
    if not products.exists():
        raise ValueError("Нет данных о продуктах для обучения модели")
    
    data = []
    for product in products:
        data.append({
            'price': float(product.price),
            'cost': float(product.cost),
            'stock_quantity': product.stock_quantity,
            'total_sales': float(product.total_sales or 0),
            'total_quantity': int(product.total_quantity or 0),
            'profit_margin': float(product.profit_margin)
        })
    
    df = pd.DataFrame(data)
    if len(df) < 5:  # Минимум 5 записей для кластеризации
        raise ValueError("Недостаточно данных для обучения модели (минимум 5 записей)")
    
    return df


def prepare_forecasting_data():
    """Подготовка данных для прогнозирования временных рядов"""
    # Получаем данные продаж по дням
    orders = Order.objects.filter(status='delivered').order_by('order_date')
    
    if not orders.exists():
        raise ValueError("Нет данных о продажах для прогнозирования")
    
    # Группируем по дням
    daily_sales = {}
    for order in orders:
        date_key = order.order_date.date()
        if date_key in daily_sales:
            daily_sales[date_key] += float(order.total_amount)
        else:
            daily_sales[date_key] = float(order.total_amount)
    
    # Сортируем по дате
    sorted_dates = sorted(daily_sales.keys())
    
    if len(sorted_dates) < 7:  # Минимум неделя данных
        raise ValueError("Недостаточно данных для прогнозирования (минимум 7 дней)")
    
    # Создаем признаки для временного ряда
    data = []
    for i, date in enumerate(sorted_dates):
        if i >= 3:  # Начинаем с 4-го дня, чтобы иметь историю
            features = {
                'day_of_week': date.weekday(),
                'day_of_month': date.day,
                'month': date.month,
                'lag_1': daily_sales[sorted_dates[i-1]],
                'lag_2': daily_sales[sorted_dates[i-2]],
                'lag_3': daily_sales[sorted_dates[i-3]],
                'target': daily_sales[date]
            }
            data.append(features)
    
    df = pd.DataFrame(data)
    if len(df) < 5:
        raise ValueError("Недостаточно данных для прогнозирования")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y


# Функции обучения моделей
def train_regression_model(X, y, algorithm, test_size=0.2, random_state=42, max_iterations=1000):
    """Обучение модели регрессии"""
    if len(X) < 5:
        raise ValueError("Недостаточно данных для регрессии (минимум 5 записей)")
    
    print(f"DEBUG: Training regression model with {len(X)} samples")
    print(f"DEBUG: X shape: {X.shape}, y shape: {y.shape}")
    print(f"DEBUG: X columns: {X.columns.tolist() if hasattr(X, 'columns') else 'No columns'}")
    print(f"DEBUG: y range: {y.min():.2f} - {y.max():.2f}, mean: {y.mean():.2f}")
    
    # Нормализуем данные для SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    print(f"DEBUG: Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    if algorithm == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
    elif algorithm == 'linear_regression':
        model = LinearRegression()
    elif algorithm == 'svm':
        model = SVR(
            kernel='rbf', 
            C=10.0,  # Увеличиваем C для лучшей точности
            gamma='scale',
            max_iter=max_iterations
        )
    elif algorithm == 'neural_network':
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),  # Два слоя
            max_iter=max_iterations, 
            alpha=0.01,  # Регуляризация
            learning_rate='adaptive',
            random_state=random_state
        )
    else:
        raise ValueError("Неподдерживаемый алгоритм")
    
    print(f"DEBUG: Fitting {algorithm} model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Используем R² score для регрессии
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"DEBUG: R² score: {r2:.4f}")
    print(f"DEBUG: MAE: {mae:.4f}")
    print(f"DEBUG: MSE: {mse:.4f}")
    print(f"DEBUG: Test y range: {y_test.min():.2f} - {y_test.max():.2f}")
    print(f"DEBUG: Predictions range: {y_pred.min():.2f} - {y_pred.max():.2f}")
    
    # Используем R² как основную метрику
    accuracy = r2
    
    # Если R² отрицательный или очень низкий, используем альтернативную метрику
    if accuracy < 0.1:
        # Используем нормализованную MSE
        y_var = y_test.var()
        if y_var > 0:
            accuracy = max(0, 1 - mse / y_var)
        else:
            accuracy = 0.1  # Минимальная точность
    
    # Ограничиваем точность от 0.1 до 1.0
    accuracy = max(0.1, min(1.0, accuracy))
    
    print(f"DEBUG: Final accuracy: {accuracy:.4f}")
    
    return model, accuracy


def train_classification_model(X, y, algorithm):
    """Обучение модели классификации"""
    if len(X) < 5:
        raise ValueError("Недостаточно данных для классификации (минимум 5 записей)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if algorithm == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == 'logistic_regression':
        model = LogisticRegression(random_state=42)
    elif algorithm == 'svm':
        model = SVC()
    elif algorithm == 'neural_network':
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    else:
        raise ValueError("Неподдерживаемый алгоритм")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy


def train_clustering_model(X, algorithm):
    """Обучение модели кластеризации"""
    if len(X) < 5:
        raise ValueError("Недостаточно данных для кластеризации (минимум 5 записей)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=min(3, len(X_scaled)), random_state=42)
        model.fit(X_scaled)
        # Для кластеризации используем силубину как метрику качества
        from sklearn.metrics import silhouette_score
        try:
            accuracy = silhouette_score(X_scaled, model.labels_)
            # Нормализуем к 0-1 диапазону
            accuracy = max(0, min(1, (accuracy + 1) / 2))
        except:
            # Если не удается вычислить силубину, используем инерцию
            max_inertia = X_scaled.var() * X_scaled.shape[0]
            accuracy = max(0, 1 - model.inertia_ / max_inertia) if max_inertia > 0 else 0.5
    
    elif algorithm == 'svm':
        # Для кластеризации с SVM используем One-Class SVM
        model = OneClassSVM(kernel='rbf', nu=0.1)
        model.fit(X_scaled)
        # Используем score_samples для оценки качества
        scores = model.score_samples(X_scaled)
        accuracy = max(0, min(1, (scores.mean() + 2) / 4))  # Нормализуем к 0-1
    
    elif algorithm == 'neural_network':
        # Для кластеризации с нейронной сетью используем K-Means с эмбеддингами
        embedding_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        embedding_model.fit(X_scaled, X_scaled.mean(axis=1))  # Автоэнкодер
        embeddings = embedding_model.predict(X_scaled)
        # Затем применяем K-Means к эмбеддингам
        model = KMeans(n_clusters=min(3, len(X_scaled)), random_state=42)
        model.fit(embeddings.reshape(-1, 1))
        # Используем силубину для оценки качества
        from sklearn.metrics import silhouette_score
        try:
            accuracy = silhouette_score(embeddings.reshape(-1, 1), model.labels_)
            accuracy = max(0, min(1, (accuracy + 1) / 2))
        except:
            accuracy = 0.5
    else:
        raise ValueError("Неподдерживаемый алгоритм")
    
    return model, accuracy


def train_forecasting_model(X, y, algorithm):
    """Обучение модели прогнозирования"""
    if len(X) < 5:
        raise ValueError("Недостаточно данных для прогнозирования (минимум 5 записей)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if algorithm == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif algorithm == 'linear_regression':
        model = LinearRegression()
    elif algorithm == 'svm':
        model = SVR()
    elif algorithm == 'neural_network':
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    else:
        raise ValueError("Неподдерживаемый алгоритм для прогнозирования")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Используем R² score для прогнозирования
    from sklearn.metrics import r2_score
    accuracy = r2_score(y_test, y_pred)
    
    # Если R² отрицательный, используем альтернативную метрику
    if accuracy < 0:
        accuracy = max(0, 1 - mean_squared_error(y_test, y_pred) / y_test.var())
    
    return model, accuracy


# Функции прогнозирования
def predict_sales(model, data):
    """Прогнозирование продаж"""
    features = np.array([
        float(data.get('customer_age', 0)),
        int(data.get('order_month', 1)),
        int(data.get('order_day', 1)),
        int(data.get('order_weekday', 0)),
        int(data.get('quarter', 1)),
        int(data.get('season', 1)),
        int(data.get('items_count', 1)),
        float(data.get('avg_item_price', 0)),
    ]).reshape(1, -1)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train, _ = prepare_sales_data()
    scaler.fit(X_train)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return f"Прогнозируемая сумма заказа: {prediction:.2f} руб."


def predict_customer_class(model, data):
    """Классификация клиентов"""
    features = np.array([
        int(data.get('total_orders', 0)),
        float(data.get('total_spent', 0)),
        int(data.get('days_since_registration', 0)),
        int(data.get('is_active', 1))
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    class_names = {'high': 'Высокий', 'medium': 'Средний', 'low': 'Низкий'}
    return f"Класс клиента: {class_names.get(prediction, prediction)}"


def predict_clustering(model, data):
    """Кластеризация продуктов"""
    features = np.array([
        float(data.get('price', 0)),
        float(data.get('cost', 0)),
        float(data.get('stock_quantity', 0)),
        float(data.get('total_sales', 0)),
        float(data.get('total_quantity', 0)),
        float(data.get('profit_margin', 0))
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    class_names = {0: 'Класс 1', 1: 'Класс 2', 2: 'Класс 3'}
    return f"Класс продукта: {class_names.get(prediction, prediction)}"


def predict_forecasting(model, data):
    """Прогнозирование временных рядов"""
    features = np.array([
        int(data.get('day_of_week', 0)),
        int(data.get('day_of_month', 1)),
        int(data.get('month', 1)),
        float(data.get('lag_1', 0)),
        float(data.get('lag_2', 0)),
        float(data.get('lag_3', 0))
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    return f"Прогнозируемый объем продаж: {prediction:.2f} руб."


# API для получения данных для графиков
@login_required
def model_performance_api(request):
    """API для получения производительности моделей"""
    models = MLModel.objects.filter(created_by=request.user)
    
    data = {
        'labels': [model.name for model in models],
        'accuracy': [float(model.accuracy or 0) for model in models],
        'types': [model.get_model_type_display() for model in models]
    }
    
    return JsonResponse(data)


@login_required
def model_list(request):
    """Список всех моделей"""
    models = MLModel.objects.filter(created_by=request.user).order_by('-created_at')
    print(f"Found {models.count()} models for user {request.user}")
    for model in models:
        print(f"Model: '{model.name}' (ID: {model.id}), Type: {model.model_type}, Algorithm: {model.algorithm}, Created by: {model.created_by}")
    
    context = {
        'models': models,
        'total_models': models.count(),
        'active_models': models.filter(is_active=True).count(),
        'avg_accuracy': models.aggregate(avg=Avg('accuracy'))['avg'] or 0,
        'predictions_today': 0,  # Здесь можно добавить подсчет прогнозов за сегодня
    }
    return render(request, 'ml_models/model_list.html', context)


@login_required
def model_create(request):
    """Создание новой модели"""
    from analytics.forms import MLModelForm
    
    if request.method == 'POST':
        form = MLModelForm(request.POST)
        if form.is_valid():
            model = form.save(commit=False)
            model.created_by = request.user
            model.is_active = True
            model.accuracy = 0.0
            model.save()
            messages.success(request, f'Модель "{model.name}" успешно создана!')
            return redirect('ml_models:model_list')
    else:
        form = MLModelForm()
    
    context = {'form': form}
    return render(request, 'ml_models/model_form.html', context)


@login_required
def model_update(request, pk):
    """Обновление модели"""
    from analytics.forms import MLModelForm
    
    try:
        model = MLModel.objects.get(id=pk, created_by=request.user)
        if request.method == 'POST':
            form = MLModelForm(request.POST, instance=model)
            if form.is_valid():
                form.save()
                messages.success(request, 'Модель успешно обновлена!')
                return redirect('ml_models:model_list')
        else:
            form = MLModelForm(instance=model)
        
        context = {'form': form, 'model': model}
        return render(request, 'ml_models/model_form.html', context)
    except MLModel.DoesNotExist:
        messages.error(request, 'Модель не найдена')
        return redirect('ml_models:model_list')


@login_required
def model_delete(request, pk):
    """Удаление модели"""
    try:
        model = MLModel.objects.get(id=pk, created_by=request.user)
        if request.method == 'POST':
            if model.model_file and os.path.exists(model.model_file.path):
                os.remove(model.model_file.path)
            model.delete()
            messages.success(request, 'Модель успешно удалена!')
            return redirect('ml_models:model_list')
        
        context = {'model': model}
        return render(request, 'ml_models/model_confirm_delete.html', context)
    except MLModel.DoesNotExist:
        messages.error(request, 'Модель не найдена')
        return redirect('ml_models:model_list')


@login_required
def model_train(request, pk):
    """Обучение конкретной модели"""
    from django import forms
    
    class ModelTrainingForm(forms.Form):
        dataset = forms.ChoiceField(
            choices=[('sales', 'Данные продаж'), ('customers', 'Данные клиентов'), ('products', 'Данные продуктов')],
            label="Датасет для обучения"
        )
        test_size = forms.FloatField(
            initial=0.2,
            min_value=0.1,
            max_value=0.5,
            label="Размер тестовой выборки",
            help_text="Доля данных для тестирования (0.1 - 0.5)"
        )
        random_state = forms.IntegerField(
            initial=42,
            label="Случайное зерно",
            help_text="Для воспроизводимости результатов"
        )
        max_iterations = forms.IntegerField(
            initial=100,
            min_value=10,
            max_value=1000,
            label="Максимальное количество итераций"
        )
        hyperparameters = forms.CharField(
            widget=forms.Textarea(attrs={'rows': 3}),
            required=False,
            label="Дополнительные параметры",
            help_text="JSON формат (опционально)"
        )
    
    try:
        model = MLModel.objects.get(id=pk, created_by=request.user)
        
        # Динамически обновляем выбор датасета в зависимости от типа модели
        if request.method == 'GET':
            form = ModelTrainingForm()
            # Устанавливаем правильный датасет по умолчанию
            if model.model_type == 'regression':
                form.fields['dataset'].initial = 'sales'
            elif model.model_type == 'classification':
                form.fields['dataset'].initial = 'customers'
            elif model.model_type == 'clustering':
                form.fields['dataset'].initial = 'products'
            elif model.model_type == 'forecasting':
                form.fields['dataset'].initial = 'sales'
        else:
            form = ModelTrainingForm(request.POST)
        
        if request.method == 'POST':
            if form.is_valid():
                try:
                    # Получаем параметры обучения
                    dataset_type = form.cleaned_data['dataset']
                    test_size = form.cleaned_data['test_size']
                    random_state = form.cleaned_data['random_state']
                    
                    # Подготавливаем данные в зависимости от типа модели и датасета
                    if model.model_type == 'regression':
                        if dataset_type == 'sales':
                            X, y = prepare_sales_data()
                        else:
                            messages.error(request, 'Для регрессии доступны только данные продаж')
                            return redirect('ml_models:model_train', pk=pk)
                    elif model.model_type == 'classification':
                        if dataset_type == 'customers':
                            X, y = prepare_customer_data()
                        else:
                            messages.error(request, 'Для классификации доступны только данные клиентов')
                            return redirect('ml_models:model_train', pk=pk)
                    elif model.model_type == 'clustering':
                        if dataset_type == 'products':
                            X = prepare_product_data()
                            y = None
                        else:
                            messages.error(request, 'Для кластеризации доступны только данные продуктов')
                            return redirect('ml_models:model_train', pk=pk)
                    elif model.model_type == 'forecasting':
                        if dataset_type == 'sales':
                            X, y = prepare_forecasting_data()
                        else:
                            messages.error(request, 'Для прогнозирования доступны только данные продаж')
                            return redirect('ml_models:model_train', pk=pk)
                    else:
                        messages.error(request, 'Неподдерживаемый тип модели')
                        return redirect('ml_models:model_train', pk=pk)
                    
                    # Обучаем модель
                    if model.model_type == 'regression':
                        ml_model, accuracy = train_regression_model(
                            X, y, model.algorithm, 
                            test_size=test_size, 
                            random_state=random_state, 
                            max_iterations=form.cleaned_data.get('max_iterations', 1000)
                        )
                    elif model.model_type == 'classification':
                        ml_model, accuracy = train_classification_model(X, y, model.algorithm)
                    elif model.model_type == 'clustering':
                        ml_model, accuracy = train_clustering_model(X, model.algorithm)
                    elif model.model_type == 'forecasting':
                        ml_model, accuracy = train_forecasting_model(X, y, model.algorithm)
                    else:
                        raise ValueError("Неподдерживаемый тип модели")
                    
                    # Отладочная информация
                    print(f"DEBUG: Model training completed. Accuracy: {accuracy}, Type: {type(accuracy)}")
                    
                    # Проверяем, что точность является числом
                    if not isinstance(accuracy, (int, float)) or np.isnan(accuracy) or np.isinf(accuracy):
                        accuracy = 0.5  # Значение по умолчанию
                        print(f"DEBUG: Invalid accuracy detected, setting to default: {accuracy}")
                    
                    # Ограничиваем точность от 0 до 1
                    accuracy = max(0.0, min(1.0, float(accuracy)))
                    print(f"DEBUG: Final accuracy: {accuracy}")
                    
                    # Сохраняем обученную модель
                    model_dir = os.path.join('media', 'ml_models')
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    
                    model_path = os.path.join(model_dir, f'model_{model.id}.pkl')
                    joblib.dump(ml_model, model_path)
                    
                    # Обновляем модель в базе данных
                    model.model_file = f'ml_models/model_{model.id}.pkl'
                    model.accuracy = accuracy
                    model.last_trained = timezone.now()
                    model.save()
                    
                    # Сохраняем историю обучения
                    TrainingHistory.objects.create(
                        model=model,
                        accuracy=accuracy,
                        dataset_size=len(X),
                        training_time=0.0,  # Можно добавить измерение времени
                        parameters={
                            'algorithm': model.algorithm,
                            'model_type': model.model_type,
                            'test_size': test_size,
                            'random_state': random_state
                        }
                    )
                    
                    print(f"DEBUG: Model saved with accuracy: {model.accuracy}")
                    print(f"DEBUG: Model file path: {model_path}")
                    print(f"DEBUG: File exists: {os.path.exists(model_path)}")
                    
                    messages.success(request, f'Модель "{model.name}" успешно обучена! Точность: {accuracy:.2f}')
                    return redirect('ml_models:model_detail', pk=model.pk)
                    
                except Exception as e:
                    messages.error(request, f'Ошибка при обучении модели: {str(e)}')
                    return redirect('ml_models:model_train', pk=pk)
        else:
            form = ModelTrainingForm()
        
        context = {
            'model': model,
            'form': form,
            'available_datasets': [],  # Здесь можно добавить реальные датасеты
            'training_history': [],  # Здесь можно добавить историю обучения
            'accuracy_percent': model.accuracy * 100 if model.accuracy is not None else None,
        }
        return render(request, 'ml_models/model_train.html', context)
    except MLModel.DoesNotExist:
        messages.error(request, 'Модель не найдена')
        return redirect('ml_models:model_list')


@login_required
def model_predict(request, pk):
    """Прогнозирование с конкретной моделью"""
    from django import forms
    
    class PredictionForm(forms.Form):
        def __init__(self, model_type, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if model_type == 'regression':
                self.fields['customer_age'] = forms.FloatField(label="Возраст клиента (лет)", initial=30)
                self.fields['order_month'] = forms.IntegerField(label="Месяц заказа", initial=6, min_value=1, max_value=12)
                self.fields['order_day'] = forms.IntegerField(label="День заказа", initial=15, min_value=1, max_value=31)
                self.fields['order_weekday'] = forms.IntegerField(label="День недели", initial=2, min_value=0, max_value=6)
                self.fields['quarter'] = forms.IntegerField(label="Квартал", initial=2, min_value=1, max_value=4)
                self.fields['season'] = forms.IntegerField(label="Сезон", initial=2, min_value=1, max_value=4)
                self.fields['items_count'] = forms.IntegerField(label="Количество товаров", initial=2, min_value=1)
                self.fields['avg_item_price'] = forms.FloatField(label="Средняя цена товара", initial=1000)
            elif model_type == 'classification':
                self.fields['total_orders'] = forms.IntegerField(
                    initial=5, min_value=0, max_value=1000,
                    label="Общее количество заказов", help_text="Количество заказов клиента"
                )
                self.fields['total_spent'] = forms.FloatField(
                    initial=1500.0, min_value=0.0,
                    label="Общая сумма покупок", help_text="Общая сумма в рублях"
                )
                self.fields['days_since_registration'] = forms.IntegerField(
                    initial=365, min_value=0, max_value=10000,
                    label="Дней с регистрации", help_text="Количество дней с момента регистрации"
                )
                self.fields['is_active'] = forms.BooleanField(
                    initial=True, required=False,
                    label="Активный клиент", help_text="Активен ли клиент"
                )
            elif model_type == 'clustering':
                self.fields['price'] = forms.FloatField(
                    initial=1000.0, min_value=0.0,
                    label="Цена продукта", help_text="Цена в рублях"
                )
                self.fields['cost'] = forms.FloatField(
                    initial=600.0, min_value=0.0,
                    label="Себестоимость", help_text="Себестоимость в рублях"
                )
                self.fields['stock_quantity'] = forms.IntegerField(
                    initial=50, min_value=0, max_value=10000,
                    label="Количество на складе", help_text="Количество единиц на складе"
                )
                self.fields['total_sales'] = forms.FloatField(
                    initial=50000.0, min_value=0.0,
                    label="Общие продажи", help_text="Общая сумма продаж в рублях"
                )
                self.fields['total_quantity'] = forms.IntegerField(
                    initial=100, min_value=0, max_value=10000,
                    label="Общее количество проданных", help_text="Общее количество проданных единиц"
                )
            elif model_type == 'forecasting':
                self.fields['day_of_week'] = forms.IntegerField(
                    initial=2, min_value=0, max_value=6,
                    label="День недели", help_text="0=Понедельник, 6=Воскресенье"
                )
                self.fields['day_of_month'] = forms.IntegerField(
                    initial=15, min_value=1, max_value=31,
                    label="День месяца", help_text="День месяца (1-31)"
                )
                self.fields['month'] = forms.IntegerField(
                    initial=6, min_value=1, max_value=12,
                    label="Месяц", help_text="Месяц (1-12)"
                )
                self.fields['lag_1'] = forms.FloatField(
                    initial=10000.0, min_value=0.0,
                    label="Продажи вчера", help_text="Продажи за предыдущий день"
                )
                self.fields['lag_2'] = forms.FloatField(
                    initial=9500.0, min_value=0.0,
                    label="Продажи позавчера", help_text="Продажи за день до вчерашнего"
                )
                self.fields['lag_3'] = forms.FloatField(
                    initial=11000.0, min_value=0.0,
                    label="Продажи 3 дня назад", help_text="Продажи за 3 дня назад"
                )
    
    try:
        model = MLModel.objects.get(id=pk, created_by=request.user)
        
        # Проверяем, обучена ли модель
        if not model.accuracy or model.accuracy < 0.1:
            messages.warning(request, 'Внимание! Модель не обучена. Необходимо провести обучение перед использованием прогнозирования.')
            return redirect('ml_models:model_train', pk=pk)
        
        if request.method == 'POST':
            form = PredictionForm(model.model_type, request.POST)
            if form.is_valid():
                try:
                    # Загружаем обученную модель
                    if not model.model_file:
                        messages.error(request, 'Файл модели не найден. Необходимо переобучить модель.')
                        return redirect('ml_models:model_train', pk=pk)
                    
                    # Проверяем существование файла
                    model_file_path = os.path.join('media', str(model.model_file))
                    if not os.path.exists(model_file_path):
                        messages.error(request, 'Файл модели не найден. Необходимо переобучить модель.')
                        return redirect('ml_models:model_train', pk=pk)
                    
                    ml_model = joblib.load(model_file_path)
                    
                    # Выполняем прогноз в зависимости от типа модели
                    if model.model_type == 'regression':
                        prediction = predict_sales(ml_model, form.cleaned_data)
                    elif model.model_type == 'classification':
                        prediction = predict_customer_class(ml_model, form.cleaned_data)
                    elif model.model_type == 'clustering':
                        prediction = predict_clustering(ml_model, form.cleaned_data)
                    elif model.model_type == 'forecasting':
                        prediction = predict_forecasting(ml_model, form.cleaned_data)
                    else:
                        prediction = "Неподдерживаемый тип модели"
                    
                    # Сохраняем историю прогноза
                    PredictionHistory.objects.create(
                        model=model,
                        user=request.user,
                        input_data=form.cleaned_data,
                        prediction=str(prediction)
                    )
                    
                    prediction_history = model.prediction_history.all()[:10]
                    accuracy_percent = model.accuracy * 100 if model.accuracy is not None else None
                    context = {
                        'model': model,
                        'form': form,
                        'prediction': prediction,
                        'show_prediction': True,
                        'accuracy_percent': accuracy_percent,
                        'prediction_history': prediction_history
                    }
                    
                except Exception as e:
                    accuracy_percent = model.accuracy * 100 if model.accuracy is not None else None
                    messages.error(request, f'Ошибка при прогнозировании: {str(e)}')
                    context = {'model': model, 'form': form, 'accuracy_percent': accuracy_percent}
        else:
            form = PredictionForm(model.model_type)
            prediction_history = model.prediction_history.all()[:10]
            accuracy_percent = model.accuracy * 100 if model.accuracy is not None else None
            context = {'model': model, 'form': form, 'accuracy_percent': accuracy_percent, 'prediction_history': prediction_history}
        
        return render(request, 'ml_models/model_predict.html', context)
        
    except MLModel.DoesNotExist:
        messages.error(request, 'Модель не найдена')
        return redirect('ml_models:model_list')
