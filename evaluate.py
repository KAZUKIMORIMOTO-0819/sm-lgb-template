import argparse
import os
import subprocess
import sys
import logging  # ロガーのインポート

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
subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    logger.info("モデルを読み込んでいます。")
    model = lgb.Booster(model_file=os.path.join('/opt/ml/model', 'model.txt'))

    logger.info("検証データを読み込んでいます。")
    val_df = pd.read_csv('/opt/ml/processing/validation/validation.csv')
    y_val = val_df.pop('target')
    X_val = val_df

    logger.info("モデルを用いて予測を行っています。")
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    logger.info(f"AUCスコア: {auc}")

    report_dict = {
        "binary_classification_metrics": {
            "auc": {
                "value": auc
            }
        }
    }

    logger.info(f"評価結果をディレクトリ {args.output_dir} に保存しています。")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluation_path = os.path.join(args.output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(report_dict, f)
    logger.info(f"評価結果が {evaluation_path} に保存されました。")
