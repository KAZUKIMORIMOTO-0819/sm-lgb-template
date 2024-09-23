import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input/data.csv')
    parser.add_argument('--train-output', type=str, default='/opt/ml/processing/train')
    parser.add_argument('--test-output', type=str, default='/opt/ml/processing/test')

    args = parser.parse_args()

    logger.info('入力データを読み込んでいます。')
    df = pd.read_csv(args.input_data,
                    engine = "python")

    logger.info('データを訓練とテストに分割しています。')
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    logger.info('訓練データを保存しています。')
    train.to_csv(os.path.join(args.train_output, 'train.csv'), index=False)

    logger.info('テストデータを保存しています。')
    test.to_csv(os.path.join(args.test_output, 'test.csv'), index=False)
