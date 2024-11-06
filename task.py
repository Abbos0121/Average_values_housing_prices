import numpy as np
import pandas as pd
from sklearn.datasets import boston_housing
from sklearn.model_selection import train_test_split

# Загрузка данных о ценах на жильё в Калифорнии
housing = boston_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Разделение данных на обучающую и тестовую выборки (80% для обучения, 20% для тестирования)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сравнение средних значений всех признаков в обучающей и тестовой выборках
mean_train = X_train.mean()
mean_test = X_test.mean()

# Создание DataFrame для удобного сравнения
mean_comparison = pd.DataFrame({
    'Train Mean': mean_train,
    'Test Mean': mean_test
})

# Вывод результатов
print("Сравнение средних значений признаков:")
print(mean_comparison)
