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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')
def get_model(config, type: str = 'LinearRegression', in_features: int = 9):
    model = ""
    if type == 'DecisionTree':
        model = DecisionTreeRegressor(random_state=42)

    elif type == 'LinearRegression':
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
        }

        grid_search = GridSearchCV(
            estimator=Ridge(max_iter=1000),
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        model = grid_search
    elif type == "randomForest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }

        rf = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=1,
        )
        model = grid_search
    elif type == "CatBoost":
        param_grid = {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5],
        }

        cb = CatBoostRegressor(random_state=42, verbose=0)

        grid_search_cb = GridSearchCV(
            estimator=cb,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=1,
        )
        model = grid_search_cb
    elif type == "Xgboost":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }

        xgb = XGBRegressor(random_state=42, eval_metric='rmse')

        grid_search_xgb = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
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

    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy()
    print(X_train)
    model.fit(X_train, y_train)

    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy()

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    best_model = model.best_estimator_

    print("Лучшие подобранные параметры:", model.best_params_)
    print("Все параметры модели:", best_model.get_params())
    print("Лучший score на кросс-валидации:", model.best_score_)
    print(f"MSE на валидации: {mse:.4f}")
    print(f"R2 на валидации: {r2:.4f}")
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'model_weights.pth'))
    else:
        joblib.dump(model, os.path.join(config.paths.checkpoint_dir, 'model_classic.pkl'))
    
    # 2. Сохраняем пайплайн (скейлер, медианы, one-hot-кодировщик)
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor_classic.pkl'))

    wandb.finish()
    print("Модель и препроцессор сохранены!")

    