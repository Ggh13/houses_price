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

from sklearn.model_selection import KFold
from models import RegressionModel

from data_prepearing import set_seed, get_transforms, data_engineering, get_data_loaders


def train(model, train_loader, val_loader, config):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.05)

    best_val_loss = float('inf')

    for epoch in range(config.training.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(config.device)
                y_batch = y_batch.to(config.device).float().unsqueeze(1)
                
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_rmse = np.sqrt(val_loss) # Если критики оптимизируют RMSE / RMSLE
        
        scheduler.step(val_loss)
        
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'best_model.pth'))

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "lr": optimizer.param_groups[0]['lr']
        })

    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Best val_rmse: {np.sqrt(best_val_loss):.4f}")

def Predict_kfold(config):
    test_path = "./data/test.csv"
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)
    n_splits = config.data.kfold.n_splits

    all_fold_preds = []

    for fold in range(1, n_splits + 1):
        pipeline_path = os.path.join(config.paths.checkpoint_dir, f"preprocessor_fold_{fold}.pkl")
        model_path = os.path.join(config.paths.checkpoint_dir, f"model_fold_{fold}.pth")

        pipeline = joblib.load(pipeline_path)
        X_test = pipeline.transform(test_df)
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.device)

        model = RegressionModel(in_features=config.in_features).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()

        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy().flatten()
            all_fold_preds.append(preds)

    mean_preds_log = np.mean(all_fold_preds, axis=0)
    final_predictions = np.expm1(mean_preds_log)

    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': final_predictions
    })
    submission.to_csv('./submission.csv', index=False)
    print("Submission saved to ./submission.csv")

def KFoldTraining(config):
    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)
    


   
    train_df = pd.read_csv(config.paths.train_csv)
    train_df = data_engineering(train_df)
    y_full = np.log1p(train_df['SalePrice'].values)

    kf = KFold(n_splits=config.data.kfold.n_splits, shuffle=True, random_state=config.seed)



    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y_full)):
        print(f"\n============ Fold {fold + 1}/{config.data.kfold.n_splits} ============")
        
        wandb.init(
            project=config.logging.wandb_project_name,
            group="kfold-experiment",
            name=f"fold_{fold + 1}",
            reinit=True
        )

        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

        

        pipeline = get_transforms(config)
        X_train = pipeline.fit_transform(train_fold_df)
        X_val = pipeline.transform(val_fold_df)

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_val = X_val.toarray()

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(np.log1p(train_fold_df['SalePrice'].values), dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(np.log1p(val_fold_df['SalePrice'].values), dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)

        in_features = X_train.shape[1]
        model = get_model(config, in_features=in_features, type='classic').to(config.device)

        train(model, train_loader, val_loader, config)

        torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, f'model_fold_{fold + 1}.pth'))
        joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, f'preprocessor_fold_{fold + 1}.pkl'))
        
        wandb.finish()

    
def get_model(config, type: str = 'classic', in_features: int = 9):
    if type == 'classic':
        return RegressionModel(in_features=in_features)
    else:
        raise ValueError(f'Invalid model type: {type}')

def ClassicTraining(config):

    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)
    
    train_loader, val_loader, pipeline = get_data_loaders(config)
    torch.cuda.empty_cache()

    model = get_model(config, type = 'classic', in_features = config.in_features)
    model = model.to(config.device)
    train(model, train_loader, val_loader, config)


    torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'model_weights.pth'))
    
    # 2. Сохраняем пайплайн (скейлер, медианы, one-hot-кодировщик)
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor.pkl'))

    wandb.finish()
    print("Модель и препроцессор сохранены!")