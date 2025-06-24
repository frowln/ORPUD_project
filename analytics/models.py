from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid


class Category(models.Model):
    """Категория продуктов"""
    name = models.CharField(max_length=100, verbose_name="Название")
    description = models.TextField(blank=True, verbose_name="Описание")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата создания")
    
    class Meta:
        verbose_name = "Категория"
        verbose_name_plural = "Категории"
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Region(models.Model):
    """Регион"""
    name = models.CharField(max_length=100, verbose_name="Название")
    code = models.CharField(max_length=10, unique=True, verbose_name="Код региона")
    population = models.IntegerField(default=0, verbose_name="Население")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата создания")
    
    class Meta:
        verbose_name = "Регион"
        verbose_name_plural = "Регионы"
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.code})"


class Customer(models.Model):
    """Клиент"""
    GENDER_CHOICES = [
        ('M', 'Мужской'),
        ('F', 'Женский'),
        ('O', 'Другой'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True, verbose_name="Пользователь")
    first_name = models.CharField(max_length=100, verbose_name="Имя")
    last_name = models.CharField(max_length=100, verbose_name="Фамилия")
    email = models.EmailField(unique=True, verbose_name="Email")
    phone = models.CharField(max_length=20, blank=True, verbose_name="Телефон")
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, blank=True, verbose_name="Пол")
    birth_date = models.DateField(null=True, blank=True, verbose_name="Дата рождения")
    region = models.ForeignKey(Region, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="Регион")
    address = models.TextField(blank=True, verbose_name="Адрес")
    registration_date = models.DateTimeField(auto_now_add=True, verbose_name="Дата регистрации")
    is_active = models.BooleanField(default=True, verbose_name="Активен")
    
    class Meta:
        verbose_name = "Клиент"
        verbose_name_plural = "Клиенты"
        ordering = ['-registration_date']
    
    def __str__(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"


class Product(models.Model):
    """Продукт"""
    name = models.CharField(max_length=200, verbose_name="Название")
    description = models.TextField(blank=True, verbose_name="Описание")
    category = models.ForeignKey(Category, on_delete=models.CASCADE, verbose_name="Категория")
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="Цена")
    cost = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="Себестоимость")
    stock_quantity = models.IntegerField(default=0, verbose_name="Количество на складе")
    sku = models.CharField(max_length=50, unique=True, verbose_name="SKU")
    image = models.ImageField(upload_to='products/', blank=True, null=True, verbose_name="Изображение")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата создания")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Дата обновления")
    is_active = models.BooleanField(default=True, verbose_name="Активен")
    
    class Meta:
        verbose_name = "Продукт"
        verbose_name_plural = "Продукты"
        ordering = ['name']
    
    def __str__(self):
        return self.name
    
    @property
    def profit_margin(self):
        """Маржинальность"""
        if self.price > 0:
            return ((self.price - self.cost) / self.price) * 100
        return 0


class Order(models.Model):
    """Заказ"""
    STATUS_CHOICES = [
        ('pending', 'Ожидает'),
        ('processing', 'В обработке'),
        ('shipped', 'Отправлен'),
        ('delivered', 'Доставлен'),
        ('cancelled', 'Отменен'),
    ]
    
    order_number = models.CharField(max_length=20, unique=True, verbose_name="Номер заказа")
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, verbose_name="Клиент")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name="Статус")
    order_date = models.DateTimeField(auto_now_add=True, verbose_name="Дата заказа")
    delivery_date = models.DateTimeField(null=True, blank=True, verbose_name="Дата доставки")
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0, verbose_name="Общая сумма")
    shipping_address = models.TextField(verbose_name="Адрес доставки")
    notes = models.TextField(blank=True, verbose_name="Примечания")
    
    class Meta:
        verbose_name = "Заказ"
        verbose_name_plural = "Заказы"
        ordering = ['-order_date']
    
    def __str__(self):
        return f"Заказ {self.order_number}"
    
    def save(self, *args, **kwargs):
        if not self.order_number:
            self.order_number = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        super().save(*args, **kwargs)


class OrderItem(models.Model):
    """Элемент заказа"""
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items', verbose_name="Заказ")
    product = models.ForeignKey(Product, on_delete=models.CASCADE, verbose_name="Продукт")
    quantity = models.IntegerField(validators=[MinValueValidator(1)], verbose_name="Количество")
    unit_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="Цена за единицу")
    total_price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="Общая цена")
    
    class Meta:
        verbose_name = "Элемент заказа"
        verbose_name_plural = "Элементы заказов"
    
    def __str__(self):
        return f"{self.product.name} x {self.quantity}"
    
    def save(self, *args, **kwargs):
        self.total_price = self.quantity * self.unit_price
        super().save(*args, **kwargs)


class MLModel(models.Model):
    """Модель машинного обучения"""
    MODEL_TYPES = [
        ('classification', 'Классификация'),
        ('regression', 'Регрессия'),
        ('clustering', 'Кластеризация'),
        ('forecasting', 'Прогнозирование'),
    ]
    
    ALGORITHM_CHOICES = [
        ('random_forest', 'Random Forest'),
        ('linear_regression', 'Linear Regression'),
        ('logistic_regression', 'Logistic Regression'),
        ('kmeans', 'K-Means'),
        ('svm', 'Support Vector Machine'),
        ('neural_network', 'Neural Network'),
    ]
    
    name = models.CharField(max_length=200, verbose_name="Название модели")
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES, verbose_name="Тип модели")
    description = models.TextField(blank=True, verbose_name="Описание")
    algorithm = models.CharField(max_length=100, choices=ALGORITHM_CHOICES, verbose_name="Алгоритм")
    accuracy = models.FloatField(null=True, blank=True, verbose_name="Точность")
    model_file = models.FileField(upload_to='ml_models/', blank=True, null=True, verbose_name="Файл модели")
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Создатель")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата создания")
    last_trained = models.DateTimeField(null=True, blank=True, verbose_name="Последнее обучение")
    is_active = models.BooleanField(default=True, verbose_name="Активна")
    
    class Meta:
        verbose_name = "Модель ML"
        verbose_name_plural = "Модели ML"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"


class DataUpload(models.Model):
    """Загрузка данных"""
    FILE_TYPES = [
        ('csv', 'CSV'),
        ('excel', 'Excel'),
        ('json', 'JSON'),
    ]
    
    name = models.CharField(max_length=200, verbose_name="Название")
    file = models.FileField(upload_to='uploads/', verbose_name="Файл")
    file_type = models.CharField(max_length=10, choices=FILE_TYPES, verbose_name="Тип файла")
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Загрузил")
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата загрузки")
    processed = models.BooleanField(default=False, verbose_name="Обработан")
    row_count = models.IntegerField(default=0, verbose_name="Количество строк")
    
    class Meta:
        verbose_name = "Загрузка данных"
        verbose_name_plural = "Загрузки данных"
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.name


class TrainingHistory(models.Model):
    """История обучения модели"""
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='training_history', verbose_name="Модель")
    training_date = models.DateTimeField(auto_now_add=True, verbose_name="Дата обучения")
    accuracy = models.FloatField(verbose_name="Точность")
    dataset_size = models.IntegerField(verbose_name="Размер датасета")
    training_time = models.FloatField(verbose_name="Время обучения (сек)")
    parameters = models.JSONField(default=dict, verbose_name="Параметры обучения")
    
    class Meta:
        verbose_name = "История обучения"
        verbose_name_plural = "История обучения"
        ordering = ['-training_date']
    
    def __str__(self):
        return f"{self.model.name} - {self.training_date.strftime('%d.%m.%Y %H:%M')}"


class Report(models.Model):
    """Отчет"""
    REPORT_TYPES = [
        ('comprehensive', 'Комплексный'),
        ('sales', 'По продажам'),
        ('products', 'По продуктам'),
        ('customers', 'По клиентам'),
    ]
    
    name = models.CharField(max_length=200, verbose_name="Название отчета")
    content = models.TextField(verbose_name="Содержание отчета")
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES, verbose_name="Тип отчета")
    generated_by = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Создатель")
    generated_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата создания")
    
    class Meta:
        verbose_name = "Отчет"
        verbose_name_plural = "Отчеты"
        ordering = ['-generated_at']
    
    def __str__(self):
        return self.name


class PredictionHistory(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='prediction_history', verbose_name="Модель")
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="Пользователь")
    input_data = models.JSONField(verbose_name="Входные данные")
    prediction = models.CharField(max_length=255, verbose_name="Результат прогноза")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата прогноза")
    
    class Meta:
        verbose_name = "История прогноза"
        verbose_name_plural = "История прогнозов"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.model.name} ({self.created_at.strftime('%d.%m.%Y %H:%M')})"
