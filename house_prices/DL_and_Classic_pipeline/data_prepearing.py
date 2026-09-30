
import random
import numpy as np
import torch
import os
import wandb


from omegaconf import DictConfig
from omegaconf import OmegaConf

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

import joblib
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold


def set_seed(seed: int):
    '''Set a random seed for complete reproducibility.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_transforms(config, df):
    ignore_cols = ['Id', 'SalePrice']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    numeric_features = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    return full_pipeline


def data_engineering(df):
    '''Создание новых признаков под House Prices.'''
    df = df.copy()
    
    if 'TotalBsmtSF' in df.columns and 'GrLivArea' in df.columns:
        df['TotalSF'] = df['TotalBsmtSF'].fillna(0) + df['GrLivArea'].fillna(0)
        
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df.columns for col in bath_cols):
        df['TotalBaths'] = (
            df['FullBath'].fillna(0) + 
            0.5 * df['HalfBath'].fillna(0) + 
            df['BsmtFullBath'].fillna(0) + 
            0.5 * df['BsmtHalfBath'].fillna(0)
        )

    return df

def get_data_loaders(config):
    df = pd.read_csv(config.paths.train_csv)
    df = data_engineering(df)

    X = df.drop(columns=['SalePrice'])
    y = np.log1p(df['SalePrice'].values)


    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.data.kfold.val_size, 
        random_state=config.seed
    )

    pipeline = get_transforms(config, df)
    

    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()


    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    return train_loader, val_loader, pipeline