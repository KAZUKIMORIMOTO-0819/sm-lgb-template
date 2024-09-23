import subprocess
import sys
import os
import tarfile

# requirements.txtのパスを修正
requirements_path = '/opt/ml/processing/input/requirements.txt'

# パッケージをインストール
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
except subprocess.CalledProcessError as e:
    print(f"パッケージのインストールに失敗しました。エラー: {e}")
    sys.exit(1)

import argparse
import os
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_model(model_dir):
    model_tar_path = os.path.join(model_dir, 'model.tar.gz')
    if os.path.exists(model_tar_path):
        logger.info(f"モデルアーカイブを解凍しています: {model_tar_path}")
        with tarfile.open(model_tar_path) as tar:
            tar.extractall(path=model_dir)
    else:
        logger.error(f"モデルアーカイブが見つかりません: {model_tar_path}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test-dir', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/evaluation')
    args = parser.parse_args()

    # モデルを解凍
    extract_model(args.model_dir)

    logger.info('モデルをロードしています。')
    model_path = os.path.join(args.model_dir, 'model.lgb')
    if not os.path.exists(model_path):
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        sys.exit(1)
    model = joblib.load(model_path)

    logger.info('テストデータを読み込んでいます。')
    test_df = pd.read_csv(os.path.join(args.test_dir, 'test.csv'))
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    logger.info('予測を行っています。')
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    logger.info('評価指標を計算しています。')
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    # 評価結果を保存
    evaluation_output_path = os.path.join(args.output_dir, 'evaluation.json')
    evaluation_dict = {'classification_metrics': {'accuracy': accuracy, 'auc': auc}}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(evaluation_output_path, 'w') as f:
        json.dump(evaluation_dict, f)
    logger.info(f"評価結果を {evaluation_output_path} に保存しました。")
