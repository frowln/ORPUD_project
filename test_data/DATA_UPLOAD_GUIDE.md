# Руководство по загрузке данных

## Обзор

Функция загрузки данных позволяет импортировать CSV, Excel и JSON файлы в систему для автоматического создания записей в базе данных.

## Поддерживаемые форматы файлов

- **CSV** (.csv) - файлы с разделителями-запятыми
- **Excel** (.xlsx, .xls) - файлы Microsoft Excel
- **JSON** (.json) - файлы в формате JSON

## Типы данных

Система автоматически определяет тип данных по названиям колонок:

### 1. Данные о клиентах
Если файл содержит колонки с ключевыми словами: `customer`, `client`, `first_name`, `last_name`, `email`

**Обязательные колонки:**
- `email` - уникальный email клиента

**Опциональные колонки:**
- `first_name` - имя
- `last_name` - фамилия
- `phone` - телефон
- `gender` - пол (M/F/O)
- `region` - регион (будет создан автоматически)
- `address` - адрес

**Пример CSV файла:**
```csv
first_name,last_name,email,phone,gender,region,address
Иван,Петров,ivan.petrov@example.com,+7-900-123-45-67,M,Москва,"ул. Ленина, д. 1"
```

### 2. Данные о продуктах
Если файл содержит колонки с ключевыми словами: `product`, `item`, `name`, `sku`, `price`

**Обязательные колонки:**
- `name` - название продукта
- `sku` - уникальный артикул

**Опциональные колонки:**
- `description` - описание
- `category` - категория (будет создана автоматически)
- `price` - цена продажи
- `cost` - себестоимость
- `stock_quantity` - количество на складе

**Пример CSV файла:**
```csv
name,description,category,price,cost,stock_quantity,sku
Ноутбук Dell XPS 13,13-дюймовый ноутбук,Электроника,89999.00,65000.00,15,SKU-LAPTOP-001
```

### 3. Данные о заказах
Если файл содержит колонки с ключевыми словами: `order`, `sale`, `customer_email`, `total_amount`

**Обязательные колонки:**
- `customer_email` - email клиента

**Опциональные колонки:**
- `customer_name` - имя клиента (если клиент не существует)
- `status` - статус заказа (pending/processing/shipped/delivered/cancelled)
- `total_amount` - общая сумма
- `shipping_address` - адрес доставки
- `notes` - примечания
- `product_sku` - артикул продукта для добавления в заказ
- `quantity` - количество товара
- `unit_price` - цена за единицу

**Пример CSV файла:**
```csv
customer_email,customer_name,total_amount,status,product_sku,quantity
ivan.petrov@example.com,Иван Петров,89999.00,delivered,SKU-LAPTOP-001,1
```

## Как использовать

### 1. Загрузка файла
1. Перейдите в раздел "Загрузка данных"
2. Нажмите "Загрузить данные"
3. Заполните форму:
   - **Название** - описание загрузки
   - **Тип файла** - выберите формат файла
   - **Файл** - выберите файл для загрузки
4. Нажмите "Загрузить"

### 2. Автоматическая обработка
После загрузки файл автоматически обрабатывается:
- Система определяет тип данных
- Создаются соответствующие записи в базе данных
- Обновляется статус загрузки

### 3. Просмотр результатов
- В списке загрузок отображается статус обработки
- Обработанные файлы помечаются зеленым значком
- Необработанные файлы можно обработать вручную

## Обработка ошибок

### Частые проблемы:
1. **Неправильный формат файла** - убедитесь, что файл соответствует поддерживаемым форматам
2. **Отсутствие обязательных колонок** - проверьте наличие необходимых полей
3. **Дублирование записей** - система использует `get_or_create` для предотвращения дублирования
4. **Неправильные типы данных** - убедитесь, что числовые поля содержат числа

### Что делать при ошибках:
1. Проверьте структуру файла
2. Убедитесь в корректности данных
3. Попробуйте обработать файл вручную через кнопку "Обработать"
4. При необходимости удалите загрузку и попробуйте снова

## Примеры файлов

В корне проекта находятся примеры файлов:
- `example_customers.csv` - пример данных о клиентах
- `example_products.csv` - пример данных о продуктах

Используйте эти файлы для тестирования функции загрузки данных.

## Безопасность

- Файлы сохраняются в папке `media/uploads/`
- Поддерживаются только безопасные форматы файлов
- Система проверяет размер файла
- Загрузки привязаны к пользователю

## Ограничения

- Максимальный размер файла: 10 МБ
- Поддерживаются только CSV, Excel и JSON форматы
- Автоматическое определение типа данных работает только для стандартных названий колонок 