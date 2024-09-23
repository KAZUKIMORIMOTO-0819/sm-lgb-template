import joblib
import os
import numpy as np
import pandas as pd
import logging
import json
import tarfile

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    logger.info("モデルをロードしています。")
    model_tar_path = os.path.join(model_dir, 'model.tar.gz')
    if os.path.exists(model_tar_path):
        logger.info(f"モデルアーカイブを解凍しています: {model_tar_path}")
        with tarfile.open(model_tar_path) as tar:
            tar.extractall(path=model_dir)
    else:
        logger.error(f"モデルアーカイブが見つかりません: {model_tar_path}")
        raise FileNotFoundError(f"モデルアーカイブが見つかりません: {model_tar_path}")
    
    model_path = os.path.join(model_dir, 'model.lgb')
    if not os.path.exists(model_path):
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type='text/csv'):
    logger.info(f"入力データを処理しています。Content type: {content_type}")
    if content_type == 'text/csv':
        data = pd.read_csv(request_body)
        return data
    elif content_type == 'application/json':
        data = pd.DataFrame(json.loads(request_body))
        return data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    logger.info("予測を行っています。")
    predictions = model.predict(input_data)
    return predictions

def output_fn(predictions, content_type='text/csv'):
    logger.info(f"予測結果を出力しています。Content type: {content_type}")
    if content_type == 'text/csv':
        result = ','.join([str(p) for p in predictions])
        return result
    elif content_type == 'application/json':
        return json.dumps(predictions.tolist())
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
