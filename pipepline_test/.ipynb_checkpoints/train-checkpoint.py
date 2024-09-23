import argparse
import os
import pandas as pd
import lightgbm as lgb
import joblib
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()

    logger.info(f"訓練データ:{os.path.join(args.train, 'train.csv')}を読み込んでいます。")
    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'))

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    logger.info('LightGBMモデルを訓練しています。')
    train_dataset = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1
    }
    model = lgb.train(params, train_dataset)

    # モデルを保存
    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, 'model.lgb')
    joblib.dump(model, path)
    logger.info(f"モデルを {path} に保存しました。")
