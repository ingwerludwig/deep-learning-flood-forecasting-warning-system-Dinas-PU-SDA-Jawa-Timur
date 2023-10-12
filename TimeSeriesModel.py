import pickle
import os
import pandas as pd
from tensorflow.keras.models import load_model
from preprocess_data import split_sequences_sliding

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


class DhompoDataPreprocessor:
    def __init__(self, model_name, model_dir, scaler_dir):
        self.model = TimeSeriesModel(
            os.path.join(model_dir, f"{model_name}.h5"),
            os.path.join(scaler_dir, f"{model_name}_x_scaler.pkl"),
            os.path.join(scaler_dir, f"{model_name}_y_scaler.pkl")
        )
    
    def preprocess_data(self, df_test, n_steps_in):
        x_test, _ = split_sequences_sliding(df_test.to_numpy(), n_steps_in, 1)
        x_test = x_test.reshape(-1, 4)
        x_test_scaled = self.model.x_scaler.transform(x_test).reshape(-1, n_steps_in, 4)
        return x_test_scaled
    
    
class PurwodadiDataPreprocessor:
    def __init__(self, model_name, model_dir, scaler_dir):
        self.model = TimeSeriesModel(
            os.path.join(model_dir, f"{model_name}.h5"),
            os.path.join(scaler_dir, f"{model_name}_x_scaler.pkl"),
            os.path.join(scaler_dir, f"{model_name}_y_scaler.pkl")
        )
    
    def preprocess_data(self, df_test, n_steps_in):
        x_test, _ = split_sequences_sliding(df_test.to_numpy(), n_steps_in, 1)
        x_test = x_test.reshape(-1, 3)
        x_test_scaled = self.model.x_scaler.transform(x_test).reshape(-1, n_steps_in, 3)
        return x_test_scaled
               
model_path = os.path.join(os.getcwd(),"Model")
scaler_path = os.path.join(os.getcwd(),"Scaler")

dhompo_gru_preprocessor = DhompoDataPreprocessor("dhompo_gru", model_path, scaler_path)
dhompo_lstm_preprocessor = DhompoDataPreprocessor("dhompo_lstm", model_path, scaler_path)
dhompo_tcn_preprocessor = DhompoDataPreprocessor("dhompo_tcn",model_path, scaler_path)
purwodadi_gru_preprocessor = PurwodadiDataPreprocessor("purwodadi_gru",model_path,scaler_path)
purwodadi_lstm_preprocessor = PurwodadiDataPreprocessor("purwodadi_lstm",model_path,scaler_path)
purwodadi_tcn_preprocessor = PurwodadiDataPreprocessor("purwodadi_tcn",model_path,scaler_path)

def get_model(model_name):
    models = {
        "dhompo_gru": dhompo_gru_preprocessor,
        "dhompo_lstm": dhompo_lstm_preprocessor,
        "dhompo_tcn": dhompo_tcn_preprocessor,
        "purwodadi_gru": purwodadi_gru_preprocessor,
        "purwodadi_lstm": purwodadi_lstm_preprocessor,
        "purwodadi_tcn": purwodadi_tcn_preprocessor
    }
    return models.get(model_name)