# House Prices Prediction

Проект для предсказания цен на жильё (Kaggle House Prices - Advanced Regression Techniques). Включает глубокое обучение (PyTorch) и классические ML-модели с автоматическим подбором гиперпараметров.

## Сравнение моделей

| Модель | MSE на валидации | R2 на валидации | RMSE на валидации |
|--------|------------------|------------------|-------------------|
| DecisionTree | 0.0333 | 0.8026 | 0.1824 |
| XGBoost | 0.0133 | 0.9211 | 0.1153 |
| CatBoost | 0.0130 | 0.9227 | 0.1141 |
| Voting | 0.0146 | 0.9137 | 0.1206 |
| RandomForest | 0.0219 | 0.8699 | 0.1481 |
| Stacking | 0.0128 | 0.9242 | 0.1130 |

**Лучший результат на лидерборде: 0.13087 (Stacking)**

## DL эксперименты

| BatchNorm \ LinearNorm | кол во нейронов | Кол - во слоев | Кол во эпох | MSE на валидации | R2 на валидации |
|------------------------|-----------------|----------------|-------------|------------------|-----------------|
| BatchNorm1d | 128 | 4 | 100 | 0.1010 | 0.4008 |
| LayerNorm | 128 | 4 | 100 | 0.1123 | 0.3337 |

| BatchNorm \ LinearNorm | Тип регуляризации | Кол - во слоев | Кол во эпох | MSE на валидации | R2 на валидации | val_rmse |
|------------------------|-------------------|----------------|-------------|------------------|-----------------|----------|
| BatchNorm1d | GELU | 4 | 100 | 0.1010 | 0.4008 | - |
| LayerNorm | GELU | 4 | 100 | 0.1123 | 0.3337 | - |
| BatchNorm1d | LeakyReLU | 4 | 100 | 0.0976 | 0.4211 | 0.3124 |
| LayerNorm | LeakyReLU | 4 | 100 | 0.0244 | 0.8555 | 0.1561 |
| BatchNorm1d | PReLU | 4 | 100 | 0.1642 | 0.4053 | 0.4053 |
| LayerNorm | PReLU | 4 | 100 | 0.0173 | 0.8971 | 0.1317 |

В обоих случаях наблюдается что MSE на валидации меньше чем на Train.

По итогу всех испытаний, на основе валидации, лучше всего демонстрирует себя Stacking. Обучу его и посмотрю результаты на лидерборде

На лидерборде получен скор: 0.13087

## Структура проекта

| Файл | Описание |
|------|----------|
| `main.py` | Точка входа, выбор режима работы |
| `config.py` | Конфигурация проекта (OmegaConf) |
| `models.py` | Нейросетевая модель (MLP с GELU) |
| `utils.py` | Тренировка DL-моделей, StratifiedKFold, inference |
| `ml_utils.py` | Классические ML-модели + GridSearch + Voting + Stacking |
| `data_prepearing.py` | Предобработка и загрузка данных |
| `requirements.txt` | Зависимости проекта |
| `draw.ipynb` | EDA-ноутбук (анализ данных) |
| `data/` | Датасет (train.csv, test.csv) |
| `checkpoints/` | Сохранённые модели и препроцессоры |
| `wandb/` | Логи экспериментов Weights & Biases |

## Конфигурация (config.py)

### Основные параметры

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `seed` | Seed для воспроизводимости | `42` |
| `mode` | Режим работы | `classic_training` |
| `type_classic` | Тип классической модели | `randomForest` |
| `device` | Устройство вычислений | `cuda` |

### Параметры обучения (training)

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `training.debug` | Режим отладки | `False` |
| `training.number_of_debug_samples` | Кол-во сэмплов в debug | `1000` |
| `training.lr` | Скорость обучения | `1e-3` |
| `training.epochs` | Количество эпох | `50` |
| `training.batch_size` | Размер батча | `32` |
| `training.weight_decay` | Коэффициент L2-регуляризации | `0.0` |
| `training.max_patience` | Эпох без улучшения до остановки | `15` |

### Параметры модели (model)

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `model.hidden_dim` | Ширина первого скрытого слоя | `32` |
| `model.num_layers` | Количество слоёв (включая выходной) | `3` |
| `model.dropout` | Вероятность dropout | `0.15` |

### Параметры scheduler (scheduler)

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `scheduler.patience` | Эпох без улучшения до снижения LR | `5` |
| `scheduler.factor` | Во сколько раз уменьшается LR | `0.1` |

### Параметры данных (data)

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `data.kfold.n_splits` | Количество фолдов | `5` |
| `data.kfold.val_size` | Доля валидации | `0.2` |
| `data.shuffle` | Перемешивать данные перед разбиением | `True` |

### Параметры путей (paths)

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `paths.checkpoint_dir` | Папка для чекпоинтов | `./checkpoints` |
| `paths.train_csv` | Путь к train.csv | `./data/train.csv` |
| `paths.test_csv` | Путь к test.csv | `./data/test.csv` |

### Параметры логирования (logging)

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `logging.wandb_project_name` | Имя проекта в W&B | `titanic-classification` |

### Параметры файлов (files)

| Параметр | Описание | Значение по умолчанию |
|----------|----------|---------------------|
| `files.data` | Папка с данными | `./data/` |
| `files.train` | Имя файла train | `train.csv` |

## Режимы работы (mode)

| Mode | Описание |
|------|----------|
| `classic_training` | Обычное обучение нейросети |
| `K_fold` | StratifiedKFold кросс-валидация DL |
| `test_inference_K_fold` | Предсказание на test (K-Fold) |
| `classic_ml` | Классические ML-модели |
| `voting_ml` | Voting-ансамбль |
| `stacking_ml` | Stacking-ансамбль |

## Доступные классические модели (type_classic)

| Ключ | Модель | Описание |
|------|--------|----------|
| `DecisionTree` | DecisionTreeRegressor | Дерево решений |
| `LinearRegression` | Ridge | L2-регрессия с GridSearch |
| `randomForest` | RandomForestRegressor | Случайный лес с GridSearch |
| `CatBoost` | CatBoostRegressor | Градиентный бустинг |
| `Xgboost` | XGBRegressor | Градиентный бустинг |

## Примеры конфигурации

### Пример 1: Обучение нейросети

```python
"mode": "classic_training",
"model": {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.3,
},
"training": {
    "lr": 1e-3,
    "epochs": 100,
    "batch_size": 64,
    "weight_decay": 1e-4,
    "max_patience": 20,
},
```

### Пример 2: XGBoost

```python
"mode": "classic_ml",
"type_classic": "Xgboost",
```

### Пример 3: Voting-ансамбль

```python
"mode": "voting_ml",
```

### Пример 4: Stacking-ансамбль

```python
"mode": "stacking_ml",
```

### Пример 5: K-Fold обучение

```python
"mode": "K_fold",
"data": {
    "kfold": {"n_splits": 5, "val_size": 0.2},
    "shuffle": True,
},
```

## Запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение нейросети
python main.py

# Режим classic_ml (в config.py установите mode: "classic_ml")
python main.py

# Режим voting_ml (в config.py установите mode: "voting_ml")
python main.py

# Режим stacking_ml (в config.py установите mode: "stacking_ml")
python main.py
```

## Предобработка данных

1. **Feature Engineering**: создание `TotalSF`, `HouseAge`, `TotalBaths`
2. **Обработка выбросов**: удаление аномалий (GrLivArea > 4000 & SalePrice < 200000)
3. **Лог-трансформация**: `LotArea`, `LotFrontage`, `TotalBsmtSF`, `GrLivArea`, `MasVnrArea`
4. **Числовые признаки**: `SimpleImputer(median)` + `StandardScaler`
5. **Категориальные признаки**: `SimpleImputer(most_frequent)` + `OneHotEncoder`
6. **Целевая переменная**: `log1p(SalePrice)` для стабилизации дисперсии
