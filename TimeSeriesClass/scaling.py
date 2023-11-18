from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from TimeSeriesClass.sliding_window import split_sequences_sliding


def scaling_data(merge_df, n_steps_in, n_steps_out):
    x_data, y_data = split_sequences_sliding(merge_df.to_numpy(), n_steps_in, n_steps_out)

    x_for_training_and_val, x_test, y_for_training_and_val, y_test = train_test_split(x_data, y_data,test_size=1 - train_percent - validation_percent)
    x_train, x_validation, y_train, y_validation = train_test_split(x_for_training_and_val, y_for_training_and_val,test_size=0.23)
    y_test = y_test[:, :, -1].reshape(-1, n_steps_out)
    y_train = y_train[:, :, -1].reshape(-1, n_steps_out)
    y_validation = y_validation[:, :, -1].reshape(-1, n_steps_out)

    x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
    x_validation_reshaped = x_validation.reshape(-1, x_validation.shape[-1])
    x_test_reshaped = x_test.reshape(-1, x_test.shape[-1])

    x_scaler = StandardScaler()
    x_scaler.fit(x_train_reshaped)

    x_train_scaled = x_scaler.transform(x_train_reshaped).reshape(x_train.shape)
    x_val_scaled = x_scaler.transform(x_validation_reshaped).reshape(x_validation.shape)
    x_test_scaled = x_scaler.transform(x_test_reshaped).reshape(x_test.shape)

    y_train_reshaped = y_train
    y_validation_reshaped = y_validation
    y_test_reshaped = y_test

    y_scaler = StandardScaler()
    y_scaler.fit(y_train_reshaped)

    y_train_scaled = y_scaler.transform(y_train_reshaped)
    y_val_scaled = y_scaler.transform(y_validation_reshaped)
    y_test_scaled = y_scaler.transform(y_test_reshaped)

    n_features = merge_df.shape[1]

    return x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, x_scaler, y_scaler, n_features
