import pickle
import os
import pandas as pd
from tensorflow.keras.models import load_model

class TimeSeriesModel:
    def __init__(self, model_path, x_scaler_path, y_scaler_path):
        self.model = load_model(model_path)
        self.x_scaler = self.load_scaler(x_scaler_path)
        self.y_scaler = self.load_scaler(y_scaler_path)
    
    def load_scaler(self, scaler_path):
        with open(scaler_path, "rb") as file:
            return pickle.load(file)
    
    def predict(self, scaled_data, y_scaler):
        predictions = y_scaler.inverse_transform(self.model.predict_generator(scaled_data).reshape(-1,1))
        predictions = pd.DataFrame(predictions)
        predictions = predictions
        predictions = pd.DataFrame(predictions)
        result = predictions[0]
        return result
            
model_path = os.path.join(os.getcwd(),"Model")
scaler_path = os.path.join(os.getcwd(),"Scaler")

dhompo_gru_model = TimeSeriesModel(os.path.join(model_path,"Dhompo_GRU_t-3.h5"), os.path.join(scaler_path,"dhompo_gru_x_scaler.pkl"), os.path.join(scaler_path,"dhompo_gru_y_scaler.pkl"))
dhompo_lstm_model = TimeSeriesModel(os.path.join(model_path,"Dhompo_LSTM_t-5.h5"), os.path.join(scaler_path,"dhompo_lstm_x_scaler.pkl"), os.path.join(scaler_path,"dhompo_lstm_y_scaler.pkl"))
dhompo_tcn_model = TimeSeriesModel(os.path.join(model_path,"Dhompo_TCN_t-5.h5"), os.path.join(scaler_path,"dhompo_tcn_x_scaler.pkl"), os.path.join(scaler_path,"dhompo_tcn_y_scaler.pkl"))
purwodadi_gru_model = TimeSeriesModel(os.path.join(model_path,"Purwodadi_GRU_t-1.h5"), os.path.join(scaler_path,"purwodadi_gru_x_scaler.pkl"), os.path.join(scaler_path,"purwodadi_gru_y_scaler.pkl"))
purwodadi_lstm_model = TimeSeriesModel(os.path.join(model_path,"Purwodadi_LSTM_t-2.h5"), os.path.join(scaler_path,"purwodadi_lstm_x_scaler.pkl"), os.path.join(scaler_path,"purwodadi_lstm_y_scaler.pkl"))
purwodadi_tcn_model = TimeSeriesModel(os.path.join(model_path,"Purwodadi_TCN_t-2.h5"), os.path.join(scaler_path,"purwodadi_tcn_x_scaler.pkl"), os.path.join(scaler_path,"purwodadi_tcn_y_scaler.pkl"))

def get_model(model_name):
    models = {
        "dhompo_gru": dhompo_gru_model,
        "dhompo_lstm": dhompo_lstm_model,
        "dhompo_tcn": dhompo_tcn_model,
        "purwodadi_gru": purwodadi_gru_model,
        "purwodadi_lstm": purwodadi_lstm_model,
        "purwodadi_tcn": purwodadi_tcn_model
    }
    return models.get(model_name)