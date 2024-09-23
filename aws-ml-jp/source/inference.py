import os
def model_fn(model_dir):
    with open(os.path.join(model_dir,'my_model.txt')) as f:
        model = f.read()[:-1]
    return model
def predict_fn(input_data, model):
    response = f'{model} for the {input_data}st time'
    return response
