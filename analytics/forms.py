from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Customer, Product, Order, Category, Region, OrderItem, MLModel


class CustomerForm(forms.ModelForm):
    class Meta:
        model = Customer
        fields = ['first_name', 'last_name', 'email', 'phone', 'gender', 
                 'birth_date', 'region', 'address']
        widgets = {
            'birth_date': forms.DateInput(attrs={'type': 'date'}),
            'address': forms.Textarea(attrs={'rows': 3}),
        }


class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'category', 'price', 'cost', 
                 'stock_quantity', 'sku', 'image']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }


class OrderItemForm(forms.ModelForm):
    class Meta:
        model = OrderItem
        fields = ['product', 'quantity']
        widgets = {
            'quantity': forms.NumberInput(attrs={'min': 1, 'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['product'].queryset = Product.objects.filter(is_active=True)


class OrderForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['customer', 'status', 'delivery_date', 'shipping_address', 'notes']
        widgets = {
            'delivery_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'shipping_address': forms.Textarea(attrs={'rows': 3}),
            'notes': forms.Textarea(attrs={'rows': 3}),
        }


class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }


class RegionForm(forms.ModelForm):
    class Meta:
        model = Region
        fields = ['name', 'code', 'population']


class DataUploadForm(forms.Form):
    name = forms.CharField(max_length=200, label="Название")
    file = forms.FileField(label="Файл")
    file_type = forms.ChoiceField(
        choices=[('csv', 'CSV'), ('excel', 'Excel'), ('json', 'JSON')],
        label="Тип файла"
    )


class MLModelForm(forms.ModelForm):
    algorithm = forms.ChoiceField(
        choices=[
            ('random_forest', 'Random Forest'),
            ('linear_regression', 'Linear Regression'),
            ('logistic_regression', 'Logistic Regression'),
            ('kmeans', 'K-Means'),
            ('svm', 'Support Vector Machine'),
            ('neural_network', 'Neural Network'),
        ],
        label="Алгоритм"
    )
    
    class Meta:
        model = MLModel
        fields = ['name', 'model_type', 'algorithm', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }


class CustomerSignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True, label="Имя")
    last_name = forms.CharField(max_length=30, required=True, label="Фамилия")
    email = forms.EmailField(max_length=254, required=True, label="Email")
    phone = forms.CharField(max_length=20, required=False, label="Телефон")
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
            # Создаем связанного клиента
            Customer.objects.create(
                user=user,
                first_name=self.cleaned_data['first_name'],
                last_name=self.cleaned_data['last_name'],
                email=self.cleaned_data['email'],
                phone=self.cleaned_data.get('phone', '')
            )
        return user 