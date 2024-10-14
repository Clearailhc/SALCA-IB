import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import DynamicLSTM
from models.cnn_model import DynamicCNN
from models.gru_model import DynamicGRU
from models.transformer_model import DynamicTransformer
from models.xgboost_model import DynamicXGBoost
from sklearn.ensemble import RandomForestClassifier

def create_model(model_name, config):
    if model_name == 'LSTM':
        return DynamicLSTM(**config)
    elif model_name == 'CNN':
        return DynamicCNN(**config)
    elif model_name == 'GRU':
        return DynamicGRU(**config)
    elif model_name == 'Transformer':
        return DynamicTransformer(**config)
    elif model_name == 'XGBoost':
        return create_xgboost_model(config)
    elif model_name == 'RandomForest':
        return RandomForestClassifier(**config)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

