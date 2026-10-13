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

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')


def get_voting_model(config, df):
    """
    Создаёт VotingRegressor из моделей, полученных через get_model.
    Каждая модель обёрнута в Pipeline с предобработкой.
    
    Args:
        config: Конфигурация проекта.
        df: DataFrame с данными (для определения типов колонок).
    
    Returns:
        VotingRegressor: Ансамбль из Ridge, RandomForest, XGBoost, CatBoost.
    """
    preprocessor = get_transforms(config, df)

    ridge_est, _ = get_model(config, type='LinearRegression')
    rf_est, _ = get_model(config, type='RandomForest')
    xgb_est, _ = get_model(config, type='XGBoost')
    cb_est, _ = get_model(config, type='CatBoost')

    voting = VotingRegressor(estimators=[
        ('ridge', Pipeline([('preprocessor', preprocessor), ('model', ridge_est)])),
        ('rf', Pipeline([('preprocessor', preprocessor), ('model', rf_est)])),
        ('xgb', Pipeline([('preprocessor', preprocessor), ('model', xgb_est)])),
        ('cb', Pipeline([('preprocessor', preprocessor), ('model', cb_est)])),
    ])

    return voting


def get_stacking_model(config, df):
    """
    Создаёт StackingRegressor из моделей, полученных через get_model.
    Базовые модели обёрнуты в Pipeline с предобработкой.
    Финальная мета-модель — LinearRegression.
    
    Args:
        config: Конфигурация проекта.
        df: DataFrame с данными (для определения типов колонок).
    
    Returns:
        StackingRegressor: Ансамбль с мета-моделью LinearRegression.
    """
    preprocessor = get_transforms(config, df)

    ridge_est, _ = get_model(config, type='LinearRegression')
    rf_est, _ = get_model(config, type='RandomForest')
    xgb_est, _ = get_model(config, type='XGBoost')
    cb_est, _ = get_model(config, type='CatBoost')

    stacking = StackingRegressor(
        estimators=[
            ('ridge', Pipeline([('preprocessor', preprocessor), ('model', ridge_est)])),
            ('rf', Pipeline([('preprocessor', preprocessor), ('model', rf_est)])),
            ('xgb', Pipeline([('preprocessor', preprocessor), ('model', xgb_est)])),
            ('cb', Pipeline([('preprocessor', preprocessor), ('model', cb_est)])),
        ],
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1
    )

    return stacking


def get_model(config, type: str = 'LinearRegression', in_features: int = 9):
    """
    Возвращает базовую модель и сетку параметров для GridSearchCV.
    Предобработка добавляется отдельно в ml_train через Pipeline,
    чтобы исключить утечку данных из validation-фолдов.
    
    Args:
        config: Конфигурация проекта.
        type: Тип модели ('DecisionTree', 'LinearRegression', 'randomForest', 'CatBoost', 'Xgboost').
        in_features: Количество входных признаков.
    
    Returns:
        tuple: (estimator, param_grid) — модель и сетка параметров.
    
    Raises:
        ValueError: Если указан неизвестный тип модели.
    """
    if type == 'DecisionTree':
        param_grid = {
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
        }
        estimator = DecisionTreeRegressor(random_state=42)

    elif type == 'LinearRegression':
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 10.0],
        }
        estimator = Ridge(max_iter=1000)

    elif type == "RandomForest":
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 5, 10, 15],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2'],
        }
        estimator = RandomForestRegressor(random_state=42)

    elif type == "CatBoost":
        param_grid = {
            'model__iterations': [100, 200, 300],
            'model__depth': [4, 6, 8],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__l2_leaf_reg': [1, 3, 5],
        }
        estimator = CatBoostRegressor(random_state=42, verbose=0)

    elif type == "XGBoost":
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0],
        }
        estimator = XGBRegressor(random_state=42, eval_metric='rmse')

    else:
        raise ValueError(f'Недопустимый тип модели: {type}')

    return estimator, param_grid


def ml_train(config):
    """
    Обучение классической модели с предобработкой внутри CV-фолдов и ранней остановкой.
    
    Выводит последние метрики (не лучшие). Если val_loss не улучшается
    более 15 эпох — обучение прекращается с уведомлением.
    
    Args:
        config: Конфигурация проекта.
    """
    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    # Загружаем сырые данные без предобработки — она будет внутри Pipeline
    X_train_raw, y_train, X_val_raw, y_val, df = get_data_loaders(config, preprocess=False)
    torch.cuda.empty_cache()

    # Вычисляем количество признаков после предобработки
    in_features = len(get_transforms(config, df).fit_transform(X_train_raw[:1]).toarray()[0]) if hasattr(get_transforms(config, df).fit_transform(X_train_raw[:1]), 'toarray') else len(get_transforms(config, df).fit_transform(X_train_raw[:1])[0])

    # Получаем базовую модель и сетку параметров
    base_estimator, param_grid = get_model(config, type=config.type_classic, in_features=in_features)

    # Создаём предобработчик (обучается внутри каждого CV-фолда)
    preprocessor = get_transforms(config, df)

    # Объединяем предобработку и модель в единый Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_estimator)
    ])

    # GridSearchCV будет обучать предобработку внутри каждого фолда
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train_raw, y_train)

    # Извлекаем лучший Pipeline и обучаем его предобработку на всём X_train
    best_pipeline = grid_search.best_estimator_
    best_pipeline.named_steps['preprocessor'].fit(X_train_raw)

    # Трансформируем X_val обученным предобработчиком
    X_val_processed = best_pipeline.named_steps['preprocessor'].transform(X_val_raw)
    if hasattr(X_val_processed, "toarray"):
        X_val_processed = X_val_processed.toarray()

    # Предсказание на внешнем holdout — используем последнюю обученную модель
    y_pred = best_pipeline.named_steps['model'].predict(X_val_processed)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mse)

    print("Лучшие подобранные параметры:", grid_search.best_params_)
    print("Все параметры модели:", best_pipeline.named_steps['model'].get_params())
    print(f"Последний MSE на валидации: {mse:.4f}")
    print(f"Последний RMSE на валидации: {rmse:.4f}")
    print(f"Последний R2 на валидации: {r2:.4f}")

    # Сохраняем обученный предобработчик и модель отдельно
    joblib.dump(best_pipeline.named_steps['preprocessor'], os.path.join(config.paths.checkpoint_dir, 'preprocessor_classic.pkl'))
    joblib.dump(best_pipeline.named_steps['model'], os.path.join(config.paths.checkpoint_dir, 'model_classic.pkl'))

    wandb.finish()
    print("Модель и препроцессор сохранены!")

    predict_submission(config)


def predict_submission_voting(config):
    """
    Генерация submission.csv для Voting-модели.
    
    Args:
        config: Конфигурация проекта.
    """
    test_path = "./data/test.csv"
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)

    model_path = os.path.join(config.paths.checkpoint_dir, 'model_voting.pkl')
    model = joblib.load(model_path)

    y_pred_log = model.predict(test_df)
    final_predictions = np.expm1(y_pred_log)

    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': final_predictions
    })
    submission.to_csv('./submission.csv', index=False)
    print("Voting submission saved to ./submission.csv")


def predict_submission_stacking(config):
    """
    Генерация submission.csv для Stacking-модели.
    
    Args:
        config: Конфигурация проекта.
    """
    test_path = "./data/test.csv"
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)

    model_path = os.path.join(config.paths.checkpoint_dir, 'model_stacking.pkl')
    model = joblib.load(model_path)

    y_pred_log = model.predict(test_df)
    final_predictions = np.expm1(y_pred_log)

    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': final_predictions
    })
    submission.to_csv('./submission.csv', index=False)
    print("Stacking submission saved to ./submission.csv")


def predict_submission(config):
    """
    Генерация submission.csv для классической модели.
    
    Args:
        config: Конфигурация проекта.
    """
    test_path = "./data/test.csv"
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)

    model_path = os.path.join(config.paths.checkpoint_dir, 'model_classic.pkl')
    pipeline_path = os.path.join(config.paths.checkpoint_dir, 'preprocessor_classic.pkl')

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    X_test = pipeline.transform(test_df)
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    y_pred_log = model.predict(X_test)
    final_predictions = np.expm1(y_pred_log)

    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': final_predictions
    })
    submission.to_csv('./submission.csv', index=False)
    print("Submission saved to ./submission.csv")


def voting_ml_train(config):
    """
    Обучение Voting-ансамбля.
    
    Обучает модели через get_model, объединяет в VotingRegressor,
    сохраняет и генерирует submission.
    
    Args:
        config: Конфигурация проекта.
    """
    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    X_train_raw, y_train, X_val_raw, y_val, df = get_data_loaders(config, preprocess=False)
    torch.cuda.empty_cache()

    model = get_voting_model(config, df)
    model.fit(X_train_raw, y_train)

    y_pred = model.predict(X_val_raw)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Voting MSE на валидации: {mse:.4f}")
    print(f"Voting RMSE на валидации: {np.sqrt(mse):.4f}")
    print(f"Voting R2 на валидации: {r2:.4f}")

    joblib.dump(model, os.path.join(config.paths.checkpoint_dir, 'model_voting.pkl'))

    wandb.finish()

    predict_submission_voting(config)


def stacking_ml_train(config):
    """
    Обучение Stacking-ансамбля.
    
    Обучает модели через get_model, объединяет в StackingRegressor,
    сохраняет и генерирует submission.
    
    Args:
        config: Конфигурация проекта.
    """
    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    X_train_raw, y_train, X_val_raw, y_val, df = get_data_loaders(config, preprocess=False)
    torch.cuda.empty_cache()

    model = get_stacking_model(config, df)
    model.fit(X_train_raw, y_train)

    y_pred = model.predict(X_val_raw)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Stacking MSE на валидации: {mse:.4f}")
    print(f"Stacking RMSE на валидации: {np.sqrt(mse):.4f}")
    print(f"Stacking R2 на валидации: {r2:.4f}")

    joblib.dump(model, os.path.join(config.paths.checkpoint_dir, 'model_stacking.pkl'))

    wandb.finish()

    predict_submission_stacking(config)
