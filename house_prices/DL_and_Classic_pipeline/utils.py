import random
import numpy as np
import torch
import os
import wandb
import warnings
from sklearn.metrics import mean_squared_error, r2_score


from omegaconf import DictConfig
from omegaconf import OmegaConf

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

import joblib
import torch.nn as nn

from sklearn.model_selection import KFold
from models import RegressionModel

from data_prepearing import set_seed, get_transforms, data_engineering, get_data_loaders


def train(model, train_loader, val_loader, config):
    """
    Обучение нейросетевой модели с валидацией, ранней остановкой и диагностикой.
    
    Выводит последние метрики (не лучшие). Если val_loss не улучшается
    более max_patience эпох — обучение прекращается с уведомлением.
    
    Args:
        model: Экземпляр RegressionModel.
        train_loader: DataLoader для обучающих данных.
        val_loader: DataLoader для валидационных данных.
        config: Конфигурация проекта.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.scheduler.patience,
        factor=config.scheduler.factor
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    max_patience = config.training.max_patience
    last_train_loss = 0.0
    last_val_loss = 0.0
    last_val_rmse = 0.0

    for epoch in range(config.training.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device).float()

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        last_train_loss = train_loss

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(config.device)
                y_batch = y_batch.to(config.device).float()

                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        last_val_loss = val_loss
        last_val_rmse = np.sqrt(val_loss)

        scheduler.step(val_loss)

        # Проверка на плато: если val_loss не улучшается более max_patience эпох — остановка
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'best_model.pth'))
        else:
            epochs_without_improvement += 1

        # Диагностика каждые 5 эпох
        if epoch % 5 == 0 or epoch == config.training.epochs - 1:
            with torch.no_grad():
                X_val, y_val = next(iter(val_loader))
                X_val = X_val.to(config.device)
                y_val = y_val.to(config.device).float()
                val_preds = model(X_val)

            print(f"\n=== Диагностика эпохи {epoch} ===")
            print(f"  train_loss: {train_loss:.6f}")
            print(f"  val_loss:   {val_loss:.6f}")
            print(f"  val_rmse:   {last_val_rmse:.6f}")
            print(f"  lr:         {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  y_batch mean: {y_val.mean().item():.4f}, std: {y_val.std().item():.4f}")
            print(f"  pred mean:    {val_preds.mean().item():.4f}, std: {val_preds.std().item():.4f}")
            print(f"  pred min/max: {val_preds.min().item():.4f} / {val_preds.max().item():.4f}")

            # Проверка градиентов первого слоя
            first_layer = model.net[0]
            if first_layer.weight.grad is not None:
                grad_norm = first_layer.weight.grad.norm().item()
                print(f"  grad norm (1-й слой): {grad_norm:.6f}")
            else:
                print(f"  grad norm (1-й слой): None (нет градиентов)")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": last_val_rmse,
            "lr": optimizer.param_groups[0]['lr']
        })

        if epochs_without_improvement >= max_patience:
            print(f"\nОбучение остановлено на эпохе {epoch + 1}: val_loss не улучшался {max_patience} эпох")
            break

    # Выводим последние метрики (не лучшие)
    print(f"\n{'='*50}")
    print(f"Последний train_loss (MSE): {last_train_loss:.4f}")
    print(f"Последний val_loss (MSE):   {last_val_loss:.4f}")
    print(f"Последний val_rmse:         {last_val_rmse:.4f}")

    # Вычисляем R2 и статистики на последней модели
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(config.device)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    val_r2 = r2_score(all_targets, all_preds)
    print(f"Последний val_r2:          {val_r2:.4f}")
    print(f"{'='*50}")

    # Статистика target и predictions
    print(f"\n{'='*50}")
    print("Статистика предсказаний на валидации:")
    print(f"  target mean: {all_targets.mean():.4f}, std: {all_targets.std():.4f}")
    print(f"  pred mean:   {all_preds.mean():.4f}, std: {all_preds.std():.4f}")
    print(f"  pred min:    {all_preds.min():.4f}, max: {all_preds.max():.4f}")
    print(f"  corr:        {np.corrcoef(all_targets, all_preds)[0, 1]:.4f}")
    print(f"{'='*50}")


def Predict_kfold(config):
    """
    Генерация submission.csv на основе усреднённых предсказаний всех фолдов.
    
    Args:
        config: Конфигурация проекта.
    """
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

        model = RegressionModel(
            in_features=X_test.shape[1],
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        ).to(config.device)
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
    """
    K-fold обучение нейросетевой модели с StratifiedKFold.
    
    Target разбивается на бины по квантилям для стратификации.
    Для каждого фолда создаётся отдельный препроцессор, обучается модель,
    сохраняются веса и препроцессор. После обучения генерируется submission.
    
    Args:
        config: Конфигурация проекта.
    """
    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    train_df = pd.read_csv(config.paths.train_csv)
    train_df = data_engineering(train_df)
    y_full = np.log1p(train_df['SalePrice'].values)

    # Разбиваем target на бины по квантилям для стратификации
    n_bins = 10
    y_binned = pd.qcut(y_full, q=n_bins, labels=False, duplicates='drop')

    skf = StratifiedKFold(
        n_splits=config.data.kfold.n_splits,
        shuffle=config.data.shuffle,
        random_state=config.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_binned)):
        print(f"\n============ Fold {fold + 1}/{config.data.kfold.n_splits} ============")

        wandb.init(
            project=config.logging.wandb_project_name,
            group="kfold-experiment",
            name=f"fold_{fold + 1}",
            reinit=True
        )

        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

        pipeline = get_transforms(config, train_fold_df)
        X_train = pipeline.fit_transform(train_fold_df)
        X_val = pipeline.transform(val_fold_df)

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_val = X_val.toarray()

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(np.log1p(train_fold_df['SalePrice'].values), dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(np.log1p(val_fold_df['SalePrice'].values), dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)

        in_features = X_train.shape[1]
        model = get_model(config, in_features=in_features, type='classic').to(config.device)

        train(model, train_loader, val_loader, config)

        torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, f'model_fold_{fold + 1}.pth'))
        joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, f'preprocessor_fold_{fold + 1}.pkl'))

        wandb.finish()

    predict_submission_dl(config, model, pipeline, use_kfold=True)


def get_model(config, type: str = 'classic', in_features: int = 9):
    """
    Фабрика моделей для DL-обучения.
    
    Args:
        config: Конфигурация проекта.
        type: Тип модели ('classic' — RegressionModel).
        in_features: Количество входных признаков.
    
    Returns:
        Экземпляр модели.
    
    Raises:
        ValueError: Если указан неизвестный тип модели.
    """
    if type == 'classic':
        return RegressionModel(
            in_features=in_features,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
    else:
        raise ValueError(f'Invalid model type: {type}')


def ClassicTraining(config):
    """
    Обычное обучение нейросетевой модели без кросс-валидации.
    
    Загружает данные, создаёт препроцессор, обучает модель,
    сохраняет веса и генерирует submission.
    
    Args:
        config: Конфигурация проекта.
    """
    wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.checkpoint_dir):
        os.makedirs(config.paths.checkpoint_dir)

    # Загружаем сырые данные без предобработки
    X_train_raw, y_train, X_val_raw, y_val, df = get_data_loaders(config, preprocess=False)
    torch.cuda.empty_cache()

    # Создаём предобработчик и обучаем его на X_train
    pipeline = get_transforms(config, df)
    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)

    in_features = X_train.shape[1]
    model = get_model(config, in_features=in_features, type='classic').to(config.device)
    train(model, train_loader, val_loader, config)

    torch.save(model.state_dict(), os.path.join(config.paths.checkpoint_dir, 'model_weights.pth'))

    # Сохраняем пайплайн (скейлер, медианы, one-hot-кодировщик)
    joblib.dump(pipeline, os.path.join(config.paths.checkpoint_dir, 'preprocessor.pkl'))

    wandb.finish()
    print("Модель и препроцессор сохранены!")

    predict_submission_dl(config, model, pipeline, use_kfold=False)


def predict_submission_dl(config, model, pipeline, use_kfold=False):
    """
    Генерация submission.csv для DL-модели (обычной или K-fold).
    
    Использует переданную обученную модель и препроцессор.
    
    Args:
        config: Конфигурация проекта.
        model: Обученная RegressionModel.
        pipeline: Обученный препроцессор.
        use_kfold: Если True, использует модели фолдов (загружает из checkpoint_dir).
                   Если False, использует переданную модель.
    """
    test_path = "./data/test.csv"
    test_df = pd.read_csv(test_path)
    test_df = data_engineering(test_df)

    if use_kfold:
        n_splits = config.data.kfold.n_splits
        all_fold_preds = []
        for fold in range(1, n_splits + 1):
            pipeline_path = os.path.join(config.paths.checkpoint_dir, f"preprocessor_fold_{fold}.pkl")
            model_path = os.path.join(config.paths.checkpoint_dir, f"model_fold_{fold}.pth")
            pipeline_fold = joblib.load(pipeline_path)
            X_test = pipeline_fold.transform(test_df)
            if hasattr(X_test, "toarray"):
                X_test = X_test.toarray()
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.device)
            model_fold = RegressionModel(
                in_features=X_test.shape[1],
                hidden_dim=config.model.hidden_dim,
                num_layers=config.model.num_layers,
                dropout=config.model.dropout
            ).to(config.device)
            model_fold.load_state_dict(torch.load(model_path, map_location=config.device))
            model_fold.eval()
            with torch.no_grad():
                preds = model_fold(X_tensor).cpu().numpy().flatten()
                all_fold_preds.append(preds)
        mean_preds_log = np.mean(all_fold_preds, axis=0)
    else:
        X_test = pipeline.transform(test_df)
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.device)
        with torch.no_grad():
            mean_preds_log = model(X_tensor).cpu().numpy().flatten()

    final_predictions = np.expm1(mean_preds_log)
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': final_predictions
    })
    submission.to_csv('./submission.csv', index=False)
    print("Submission saved to ./submission.csv")
