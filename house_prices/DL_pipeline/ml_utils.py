import random
import numpy as np
import torch
import os
import wandb


from omegaconf import DictConfig
from omegaconf import OmegaConf

import pandas as pd

import joblib
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold

from data_prepearing import set_seed, get_transforms, data_engineering, get_data_loaders

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
def get_model(config, type: str = 'LogisticRegression', in_features: int = 9):
    model = ""
    if type == 'DecisionTree':
        model = DecisionTreeClassifier()  

    elif type == 'LogisticRegression':
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear', 'saga']
        }

        grid_search = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000),
            param_grid=param_grid,
            cv=5,               
            scoring='accuracy', 
            n_jobs=-1           
        )
        model = grid_search
    elif type == "randomForest":
        param_grid = {
            'n_estimators': [50, 100, 200],  # Количество деревьев в лесу
            'max_depth': [None, 5, 10, 15],  # Максимальная глубина деревьев
            'min_samples_split': [
                2,
                5,
                10,
            ],  # Минимальное число объектов для разбиения узла
            'min_samples_leaf': [1, 2, 4],  # Минимальное число объектов в листе
            'max_features': ['sqrt', 'log2'],  # Число признаков для выбора при сплите
        }

        # 2. Инициализируем базовую модель
        rf = RandomForestClassifier(random_state=42)

        # 3. Настраиваем GridSearch
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,  # 5-фолдовая кросс-валидация
            scoring='accuracy',
            n_jobs=1,  # Используем 1 поток во избежание проблем с памятью на Windows
        )
        model = grid_search
    elif type == "CatBoost":
        param_grid = {
            'iterations': [100, 200, 300],  # Аналог n_estimators (количество деревьев)
            'depth': [4, 6, 8],  # Глубина деревьев (обычно 4-6 оптимально)
            'learning_rate': [0.01, 0.05, 0.1],  # Скорость обучения
            'l2_leaf_reg': [1, 3, 5],  # L2-регуляризация (защита от переобучения)
        }

        # 2. Инициализация (verbose=0 выключает спам логов в консоль)
        cb = CatBoostClassifier(random_state=42, verbose=0)

        # 3. GridSearch
        grid_search_cb = GridSearchCV(
            estimator=cb,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1,  # Защита от ошибки с памятью на Windows
        )
        model = grid_search_cb
    elif type == "Xgboost":
        param_grid = {
            'n_estimators': [50, 100, 200],  # Количество деревьев
            'max_depth': [3, 5, 7],  # Глубина деревьев
            'learning_rate': [0.01, 0.05, 0.1],  # Скорость обучения
            'subsample': [0.8, 1.0],  # Доля выборки для построения дерева
            'colsample_bytree': [0.8, 1.0],  # Доля признаков для каждого дерева
        }

        # 2. Инициализация
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')

        # 3. GridSearch
        grid_search_xgb = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
        )
        model = grid_search_xgb
    else:
        raise ValueError(f'Invalid model type: {type}')
    return model

def ml_train(config):
    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)
    
    train_loader, val_loader, pipeline = get_data_loaders(config)
    torch.cuda.empty_cache()

    model = get_model(config, type = 'Xgboost', in_features = config.in_features)
    #model = model.to(config.device)

    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy()
    print(X_train)
    model.fit(X_train, y_train)


    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy()

    model.fit(X_train, y_train)
    y_pred=model.predict(X_val)
    #print(accuracy_score(y_val,y_pred))
    best_model = model.best_estimator_

    # 1. Вывести только те параметры, которые подбирались через сетку (param_grid)
    print("Лучшие подломанные параметры:", model.best_params_)

    # 2. Вывести абсолютно все параметры объекта лучшей модели
    print("Все параметры модели:", best_model.get_params())

    # 3. (Опционально) Вывести средний результат (Accuracy) лучшей модели на кросс-валидации
    print("Лучший score на кросс-валидации:", model.best_score_)
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'model_weights.pth'))
    else:
        joblib.dump(model, os.path.join(config.paths.checkpoint_dir, 'model_classic.pkl'))
    
    # 2. Сохраняем пайплайн (скейлер, медианы, one-hot-кодировщик)
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor_classic.pkl'))

    wandb.finish()
    print("Модель и препроцессор сохранены!")

    