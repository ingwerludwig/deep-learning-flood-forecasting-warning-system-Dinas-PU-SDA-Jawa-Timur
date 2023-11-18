import pickle
import os
import pandas as pd
from tensorflow.keras.models import load_model
from keras import backend as K
from tensorflow.keras.layers import LSTM, GRU
from tcn import TCN
import warnings

warnings.filterwarnings("ignore")


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def load_scaler(path_to_scaler):
    with open(path_to_scaler, "rb") as file:
        return pickle.load(file)


class TimeSeriesModel:
    def __init__(self, path_to_model, x_scaler_path, y_scaler_path):
        self.model = load_model(path_to_model,
                                custom_objects={'root_mean_squared_error': root_mean_squared_error, 'TCN': TCN,
                                                'LSTM': LSTM, 'GRU': GRU})
        self.x_scaler = load_scaler(x_scaler_path)
        self.y_scaler = load_scaler(y_scaler_path)

    def predict(self, scaled_data, y_scaler, n_steps_out):
        predictions = y_scaler.inverse_transform(self.model.predict_generator(scaled_data).reshape(-1, n_steps_out))
        predictions = pd.DataFrame(predictions)
        predictions = predictions
        predictions = pd.DataFrame(predictions)
        return predictions


class DhompoDataPreprocessor:
    def __init__(self, model_name, model_dir, scaler_dir):
        self.model = TimeSeriesModel(
            os.path.join(model_dir, f"{model_name}.h5"),
            os.path.join(scaler_dir, f"{model_name}_x_scaler.pkl"),
            os.path.join(scaler_dir, f"{model_name}_y_scaler.pkl")
        )

    def preprocess_data(self, df_test, n_steps_in):
        df = df_test
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.sort_values(by='DateTime', ascending=False)
        df = df.drop(columns=['DateTime'])

        x_test_scaled = self.model.x_scaler.transform(df.to_numpy()).reshape(-1, n_steps_in, 4)
        return x_test_scaled


class PurwodadiDataPreprocessor:
    def __init__(self, model_name, model_dir, scaler_dir):
        self.model = TimeSeriesModel(
            os.path.join(model_dir, f"{model_name}.h5"),
            os.path.join(scaler_dir, f"{model_name}_x_scaler.pkl"),
            os.path.join(scaler_dir, f"{model_name}_y_scaler.pkl")
        )

    def preprocess_data(self, df_test, n_steps_in):
        df = df_test
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.sort_values(by='DateTime', ascending=False)
        df = df.drop(columns=['DateTime', 'LD'])

        x_test_scaled = self.model.x_scaler.transform(df.to_numpy()).reshape(-1, n_steps_in, 3)
        return x_test_scaled


model_path = os.path.join(os.getcwd(), "model")
scaler_path = os.path.join(os.getcwd(), "scaler")

dhompo_gru_preprocessor = DhompoDataPreprocessor("dhompo_gru", model_path, scaler_path)
dhompo_lstm_preprocessor = DhompoDataPreprocessor("dhompo_lstm", model_path, scaler_path)
dhompo_tcn_preprocessor = DhompoDataPreprocessor("dhompo_tcn", model_path, scaler_path)
purwodadi_gru_preprocessor = PurwodadiDataPreprocessor("purwodadi_gru", model_path, scaler_path)
purwodadi_lstm_preprocessor = PurwodadiDataPreprocessor("purwodadi_lstm", model_path, scaler_path)
purwodadi_tcn_preprocessor = PurwodadiDataPreprocessor("purwodadi_tcn", model_path, scaler_path)


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
