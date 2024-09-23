
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from io import StringIO

def model_fn(model_dir):
    print("モデルを読み込んでいます。")
    model = lgb.Booster(model_file=os.path.join(model_dir, 'model.txt'))
    return model

def input_fn(request_body, content_type):
    if content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body), header=None)
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, accept):
    if accept == 'text/csv':
        return ','.join(map(str, prediction.tolist()))
    else:
        raise ValueError("Unsupported accept type: {}".format(accept))
