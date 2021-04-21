import joblib
from pathlib import Path
import dienen

def load_weights(weights, keras_model,has_variable_names=True):
    if isinstance(weights,str) or isinstance(weights,Path):
        weights = joblib.load(weights)

    for layer_name,weight_i in weights.items():
        if has_variable_names:
            weight_i = weight_i[0]
        keras_model.get_layer(layer_name).set_weights(weight_i)

def load_model(filename):
    model_data = joblib.load(filename)
    dienen_model = dienen.Model(model_data['original_config'])
    model_data.pop('original_config')
    dienen_model.build()
    if 'weights' in model_data:
        load_weights(model_data['weights'],dienen_model.core_model.model)
        model_data.pop('weights')
    if 'optimizer_state' in model_data:
        dienen_model.set_optimizer_weights(model_data['optimizer_state'])
        model_data.pop('optimizer_state')
    dienen_model.set_metadata(model_data)

    return dienen_model


