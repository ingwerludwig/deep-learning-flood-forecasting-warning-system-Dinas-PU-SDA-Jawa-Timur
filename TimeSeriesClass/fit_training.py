import pickle
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def train_data(model_time_series, x_data_scaled, n_features, model_daerah):
    try:
        epoch = 50
        model_time_series.fit(x_data_scaled, batch_size=64, epochs=epoch)
        model_time_series.save(f"{model_daerah}.h5")

        return "successful"
    except Exception as e:
        print(f"Error during training: {e}")
        return "failed"
