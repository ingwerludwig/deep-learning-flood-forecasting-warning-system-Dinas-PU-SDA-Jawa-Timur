from sklearn.preprocessing import MinMaxScaler
from sliding_window import split_sequences_sliding


def scaling_data(df_train, df_validation, df_test, n_steps_in, n_steps_out):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train, y_train = split_sequences_sliding(df_train.to_numpy(), n_steps_in, n_steps_out)
    x_train = x_train.reshape(-1, df_test.shape[1])
    y_train = y_train[:, :, -1].reshape(-1, n_steps_out)

    x_scaler.fit(x_train)
    y_scaler.fit(y_train)
    x_train_scaled = x_scaler.transform(x_train).reshape(-1, n_steps_in, df_test.shape[1])
    y_train_scaled = y_scaler.transform(y_train)

    x_val, y_val = split_sequences_sliding(df_validation.to_numpy(), n_steps_in, n_steps_out)
    x_val = x_val.reshape(-1, df_test.shape[1])
    y_val = y_val[:, :, -1].reshape(-1, n_steps_out)
    x_val_scaled = x_scaler.transform(x_val).reshape(-1, n_steps_in, df_test.shape[1])
    y_val_scaled = y_scaler.transform(y_val)

    x_test, y_test = split_sequences_sliding(df_test.to_numpy(), n_steps_in, n_steps_out)
    x_test = x_test.reshape(-1, df_test.shape[1])
    y_test = y_test[:, :, -1].reshape(-1, n_steps_out)
    x_test_scaled = x_scaler.transform(x_test).reshape(-1, n_steps_in, df_test.shape[1])
    y_test_scaled = y_scaler.transform(y_test)

    n_features = x_train.shape[1]

    return x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, x_scaler, y_scaler, n_features
