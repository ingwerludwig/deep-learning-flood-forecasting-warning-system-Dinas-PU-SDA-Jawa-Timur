from TimeSeriesClass.train_test_split import train_test_split_data
from TimeSeriesClass.scaling import scaling_data
from TimeSeriesClass.fit_training import train_data
from TimeSeriesClass.get_data import get_data_for_train
from TimeSeriesClass.timeseries import get_model
import logging


def get_n_steps_in_for_model(model_name):
    model_to_n_steps = {
        "dhompo_gru": 5,
        "dhompo_lstm": 5,
        "dhompo_tcn": 5,
        "purwodadi_gru": 3,
        "purwodadi_lstm": 3,
        "purwodadi_tcn": 3,
    }

    if model_name in model_to_n_steps:
        return model_to_n_steps[model_name]
    else:
        return 0


def get_n_steps_out_for_model(model_name):
    model_to_n_steps = {
        "dhompo_gru": 5,
        "dhompo_lstm": 5,
        "dhompo_tcn": 5,
        "purwodadi_gru": 3,
        "purwodadi_lstm": 3,
        "purwodadi_tcn": 3,
    }

    if model_name in model_to_n_steps:
        return model_to_n_steps[model_name]
    else:
        return 0


def training_init(x_scaler, y_scaler, n_steps_in, n_steps_out, model_time_series, merge_df, model_daerah):
    (x_data_scaled, n_features) = scaling_data(x_scaler, y_scaler, n_steps_in, n_steps_out, merge_df)
    result = train_data(model_time_series, x_data_scaled, n_features, model_daerah)

    return result


def process_model(x_scaler, y_scaler, n_steps_in, n_steps_out, model_time_series, merge_df, model_daerah):
    result = training_init(x_scaler, y_scaler, n_steps_in, n_steps_out, model_time_series, merge_df, model_daerah)
    return model_daerah, result


def start():
    arr_model_daerah = [
        "dhompo_gru",
        "dhompo_lstm",
        "dhompo_tcn",
        "purwodadi_gru",
        "purwodadi_lstm",
        "purwodadi_tcn"
    ]

    merge_df = get_data_for_train()

    results = []
    for model_daerah in arr_model_daerah:
        n_steps_in = get_n_steps_in_for_model(model_daerah)
        n_steps_out = get_n_steps_out_for_model(model_daerah)
        select_model = get_model(model_daerah)
        y_scaler = select_model.model.y_scaler
        x_scaler = select_model.model.x_scaler
        model_time_series = select_model.model
        print(model_time_series.model)

        result = process_model(x_scaler, y_scaler, n_steps_in, n_steps_out, model_time_series, merge_df, model_daerah)
        results.append(result)

    model_dict = {model: result for model, result in results}
    print(model_dict)


if __name__ == '__main__':
    start()
