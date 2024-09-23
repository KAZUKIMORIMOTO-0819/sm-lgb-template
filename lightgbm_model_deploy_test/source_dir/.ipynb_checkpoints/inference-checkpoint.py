import os
import lightgbm as lgb
import json
import logging
import json
import tarfile

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def model_fn(model_dir):
    
    model_path = os.path.join(model_dir, 'model.txt')
    model = lgb.Booster(model_file=model_path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data['data']
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction.tolist()

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps({'predictions': prediction})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
