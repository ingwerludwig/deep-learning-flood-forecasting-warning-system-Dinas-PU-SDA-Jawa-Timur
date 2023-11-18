from TimeSeriesClass.train_test_split import train_test_split_data
from TimeSeriesClass.scaling import scaling_data
from TimeSeriesClass.model_definition import model_init
from TimeSeriesClass.fit_training import train_data
from TimeSeriesClass.get_data import get_data_for_train
from apscheduler.schedulers.blocking import BlockingScheduler
import logging

logging.basicConfig(filename='scheduler.log', level=logging.INFO, format='%(asctime)s - %(message)s')
scheduler = BlockingScheduler()


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


def training_init(merge_df, n_steps_in, n_steps_out, dearah_model):
    (x_train_scaled, y_train_scaled,
     x_val_scaled, y_val_scaled,
     x_test_scaled, y_test_scaled,
     x_scaler, y_scaler,
     n_features) = scaling_data(merge_df, n_steps_in, n_steps_out)

    model_time_series = model_init(n_steps_in, n_steps_out, n_features)
    result = train_data(model_time_series, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_scaler,
                        y_scaler, dearah_model)

    return result


def process_model(model_daerah, n_steps_in, n_steps_out, merge_df):
    result = training_init(merge_df, n_steps_in, n_steps_out, model_daerah)
    return model_daerah, result


@scheduler.scheduled_job('interval', minutes=50)
def start():
    arr_model_daerah = [
        "dhompo_gru",
        "dhompo_lstm",
        "purwodadi_gru",
        "purwodadi_lstm"
    ]

    merge_df = get_data_for_train()

    results = []
    for model_daerah in arr_model_daerah:
        n_steps_in = get_n_steps_in_for_model(model_daerah)
        n_steps_out = get_n_steps_out_for_model(model_daerah)
        result = process_model(model_daerah, n_steps_in, n_steps_out, merge_df)
        results.append(result)

    model_dict = {model: result for model, result in results}
    print(model_dict)


if __name__ == '__main__':
    start()
