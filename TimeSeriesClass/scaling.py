from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from TimeSeriesClass.sliding_window import split_sequences_sliding
import numpy as np


def scaling_data(x_scaler, y_scaler, n_steps_in, n_steps_out, merge_df):
    x_data, y_data = split_sequences_sliding(merge_df.to_numpy(), n_steps_in, n_steps_out)

    x_train = x_data.reshape(merge_df.shape[0], n_steps_out, 4)
    x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
    x_data_scaled = x_scaler.transform(x_train_reshaped).reshape(merge_df.shape[0], n_steps_out, 4)

    n_features = 4

    return x_data_scaled, n_features

