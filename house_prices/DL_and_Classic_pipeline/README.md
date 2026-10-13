# House Prices Prediction

Проект для предсказания цен на жильё (Kaggle House Prices - Advanced Regression Techniques). Включает как глубокое обучение (PyTorch), так и классические ML-модели с автоматическим подбором гиперпараметров.



## Сравнение моделей

| Модель          | MSE на валидации | R2 на валидации | RMSE на валидации | 
|-----------------|---------------------|-------------------|----------|
| DecisionTree    | 0.0333             | 0.8026          |  0.1824      |  
| XGBoost         | 0.0133           | 0.9211             |   0.1153      |
| CatBoost        | 0.0130             | 0.9227            |    0.1141     |
| voting_ml        | 0.0146            | 0.9137           |     0.1206     |
| RandomForest    | 0.0219            | 0.8699           |   0.1481          |
| stacking_ml      |  0.0128         | 0.9242           |     0.1130      |


DL эксперименты

| BatchNorm \ LinearNorm | кол во нейронов | Кол - во слоев | Кол во эпох      | MSE на валидации | R2 на валидации | 
|------------------------|-----------------|----------------|------------------|-----------------|-----------------|
| BatchNorm1d            | 128            |     4           |  100              |  0.1010       |  0.4008    |   
| LayerNorm            | 128            |     4           |  100              |  0.1123       |  0.3337    |


| BatchNorm \ LinearNorm | Тип регуляризации | Кол - во слоев | Кол во эпох      | MSE на валидации | R2 на валидации | val_rmse|
|------------------------|-----------------|----------------|------------------|-----------------|-----------------|-----------|
| BatchNorm1d            | GELU             |     4           |  100              |  0.1010       |  0.4008    |    - |
| LayerNorm              | GELU            |     4           |  100              |  0.1123       |  0.3337    | -  |
| BatchNorm1d            | LeakyReLU             |     4           |  100              | 0.0976       |  0.4211    |    0.3124 |
| LayerNorm              | LeakyReLU            |     4           |  100              | 0.0244       |  0.8555    |  0.1561 |
| BatchNorm1d            | PReLU             |     4           |  100              |  0.1642      |  0.4053    |  0.4053  |
| LayerNorm              | PReLU            |     4           |  100              | 0.0173      | 0.8971    | 0.1317 |


В обоих случаях наблюдается что MSE на валидации меньше чем на Train.

По итогу всех испытаний, на основе валидации, лучше всего демонстрирует себя Stacking. Обучу его и посмотрю результаты на лидерборде
На лидерборде получен скор: 0.13087

## Структура проекта

| Файл | Описание |
|------|----------|
| `main.py` | Точка входа, выбор режима работы |
| `config.py` | Конфигурация проекта (OmegaConf) |
| `models.py` | Нейросетевая модель (MLP) |
| `utils.py` | Тренировка DL-моделей, K-Fold, inference |
| `ml_utils.py` | Классические ML-модели + GridSearch |
| `data_prepearing.py` | Предобработка и загрузка данных |
| `requirements.txt` | Зависимости проекта |
| `draw.ipynb` | EDA-ноутбук (анализ данных) |
| `data/` | Датасет (train.csv, test.csv) |
| `checkpoints/` | Сохранённые модели и препроцессоры |
| `wandb/` | Логи экспериментов Weights & Biases |

## Параметры конфигурации

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `seed` | Seed для воспроизводимости | `42` |
| `mode` | Режим работы | `classic_training` |
| `device` | Устройство вычислений | `cuda` |
| `in_features` | Количество входных признаков | `288` |
| `training.debug` | Режим отладки | `False` |
| `training.number_of_debug_samples` | Кол-во сэмплов в debug | `1000` |
| `training.lr` | Скорость обучения | `1e-3` |
| `training.epochs` | Количество эпох | `60` |
| `training.batch_size` | Размер батча | `32` |
| `data.kfold.use_kfold` | Использовать K-Fold | `True` |
| `data.kfold.n_splits` | Количество фолдов | `6` |
| `data.kfold.val_size` | Доля валидации | `0.2` |
| `paths.checkpoint_dir` | Папка для чекпоинтов | `./checkpoints` |
| `paths.train_csv` | Путь к train.csv | `./data/train.csv` |
| `logging.wandb_project_name` | Имя проекта в W&B | `titanic-classification` |

## Режимы работы

| Mode | Описание |
|------|----------|
| `classic_training` | Обычное обучение нейросети |
| `K_fold` | K-Fold кросс-валидация DL |
| `test_inference_K_fold` | Предсказание на test (K-Fold) |
| `classic_ml` | Классические ML-модели |

## Режим classic_ml

### Доступные модели

| Модель | Ключ | Описание |
|--------|------|----------|
| LogisticRegression | `LogisticRegression` | Логистическая регрессия с GridSearch |
| DecisionTree | `DecisionTree` | Дерево решений |
| RandomForest | `randomForest` | Случайный лес с GridSearch |
| CatBoost | `CatBoost` | Градиентный бустинг (CatBoost) |
| XGBoost | `Xgboost` | Градиентный бустинг (XGBoost) |

### Как сменить модель

В `ml_utils.py` в функции `ml_train` измените параметр `type`:

```python
model = get_model(config, type='Xgboost', in_features=config.in_features)
# Допустимые значения: 'LogisticRegression', 'DecisionTree', 'randomForest', 'CatBoost', 'Xgboost'
```

## Запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение нейросети
python main.py

# Режим classic_ml (в config.py установите mode: "classic_ml")
python main.py
```

## Предобработка данных

1. **Feature Engineering**: создание `TotalSF`, `HouseAge`, `TotalBaths`
2. **Числовые признаки**: `SimpleImputer(median)` + `StandardScaler`
3. **Категориальные признаки**: `SimpleImputer(most_frequent)` + `OneHotEncoder`
4. **Целевая переменная**: `log1p(SalePrice)` для стабилизации дисперсии
