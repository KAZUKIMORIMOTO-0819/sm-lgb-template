import argparse
import os
import subprocess
import sys
import logging  # loggingモジュールをインポート

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,  # ログレベルをINFOに設定
    format="%(asctime)s [%(levelname)s] %(message)s",  # ログのフォーマット
    handlers=[
        logging.StreamHandler(sys.stdout)  # 標準出力にログを表示
    ]
)
logger = logging.getLogger(__name__)

# 必要なパッケージをインストール
subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "optuna"])

import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
    }

    logger.info(f"Training with parameters: {params}")

    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(params, train_dataset, valid_sets=[val_dataset])
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    logger.info(f"Trial result: AUC = {auc}")
    return auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    logger.info("訓練データを読み込んでいます。")
    train_df = pd.read_csv('/opt/ml/input/data/train/train.csv')

    logger.info("検証データを読み込んでいます。")
    val_df = pd.read_csv('/opt/ml/input/data/validation/validation.csv')

    y_train = train_df.pop('target')
    X_train = train_df
    y_val = val_df.pop('target')
    X_val = val_df

    # Optunaによるハイパーパラメータ最適化
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), 
    n_trials=args.n_trials)

    logger.info('Best trial:')
    trial = study.best_trial
    logger.info('  AUC: {}'.format(trial.value))
    logger.info('  Params: ')
    for key, value in trial.params.items():
        logger.info('    {}: {}'.format(key, value))

    # 最適なハイパーパラメータでモデルを再学習
    best_params = trial.params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['verbosity'] = -1
    best_params['boosting_type'] = 'gbdt'

    logger.info(f"Retraining model with best parameters: {best_params}")

    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(best_params, train_dataset, valid_sets=[val_dataset])

    # モデルを保存
    model_dir = os.environ.get('SM_MODEL_DIR', './')
    model.save_model(os.path.join(model_dir, "model.txt"))
    logger.info(f"Model saved to {os.path.join(model_dir, 'model.txt')}")
