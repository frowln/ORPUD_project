# Аналитическая платформа с машинным обучением

Веб-платформа для анализа данных с предиктивной аналитикой и машинным обучением.

## Функциональность

- 📊 Загрузка и анализ данных
- 🤖 Машинное обучение и предиктивная аналитика
- 📈 Интерактивные графики и визуализация
- 🔐 Система авторизации с разными уровнями доступа
- 📋 CRUD операции для всех сущностей
- 🎯 Дэшборд с ключевыми показателями

## Технологии

- **Backend**: Django 4.2.7
- **Frontend**: Bootstrap 5, Plotly.js
- **ML**: scikit-learn, pandas, numpy
- **Визуализация**: Plotly, Matplotlib, Seaborn
- **База данных**: SQLite (можно заменить на PostgreSQL)

## Установка

1. Клонируйте репозиторий
2. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # или
   venv\Scripts\activate  # Windows
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Выполните миграции:
   ```bash
   python manage.py migrate
   ```
5. Создайте суперпользователя:
   ```bash
   python manage.py createsuperuser
   ```
6. Запустите сервер:
   ```bash
   python manage.py runserver
   ```

## Структура проекта

- `analytics/` - основное приложение
- `ml_models/` - модуль машинного обучения
- `dashboard/` - дэшборд и аналитика
- `data_management/` - управление данными
- `static/` - статические файлы
- `templates/` - HTML шаблоны

## Модели данных

1. **User** - пользователи системы
2. **Customer** - клиенты
3. **Product** - продукты
4. **Order** - заказы
5. **OrderItem** - элементы заказов
6. **Category** - категории продуктов
7. **Region** - регионы
8. **MLModel** - обученные модели ML 