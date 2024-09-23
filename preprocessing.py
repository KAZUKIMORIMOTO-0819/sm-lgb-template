import argparse
import os
import pandas as pd
import boto3
import logging
from sklearn.model_selection import train_test_split
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("前処理を開始します。")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", type=str)
    parser.add_argument("--input-key", type=str)
    args = parser.parse_args()

    logger.info("S3からデータを読み込んでいます。")
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=args.input_bucket, Key=args.input_key)
    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

    logger.info(f"入力データの形状: {df.shape}")

    logger.info("データを訓練セットと検証セットに分割します。")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs('/opt/ml/processing/train', exist_ok=True)
    os.makedirs('/opt/ml/processing/validation', exist_ok=True)

    logger.info("訓練データを保存します。")
    train_df.to_csv('/opt/ml/processing/train/train.csv', index=False)

    logger.info("検証データを保存します。")
    val_df.to_csv('/opt/ml/processing/validation/validation.csv', index=False)

    logger.info("前処理が完了しました。")