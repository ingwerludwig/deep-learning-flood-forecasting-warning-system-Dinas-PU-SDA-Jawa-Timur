import pandas as pd
from train_test_split import train_test_split_data
from scaling import scaling_data
from model_definition import model_init
from fit_training import train_data
from get_data import get_data_for_train


def get_n_steps_in_for_model(model_name):
    model_to_n_steps = {
        "dhompo_gru": 3,
        "dhompo_lstm": 5,
        "dhompo_tcn": 5,
        "purwodadi_gru": 1,
        "purwodadi_lstm": 2,
        "purwodadi_tcn": 2,
    }

    if model_name in model_to_n_steps:
        return model_to_n_steps[model_name]
    else:
        return 0


def training_init(merge_df, n_steps_in, dearah_model):
    n_steps_out = 1
    df_train, df_validation, df_test = train_test_split_data(merge_df)
    (x_train_scaled, y_train_scaled,
     x_val_scaled, y_val_scaled,
     x_test_scaled, y_test_scaled,
     x_scaler, y_scaler,
     n_features) = scaling_data(df_train, df_validation, df_test, n_steps_in, n_steps_out)

    model_time_series = model_init(n_steps_in, n_steps_out, n_features)
    result = train_data(model_time_series, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_scaler,
                        y_scaler, dearah_model)

    return result


def process_model(model_daerah, n_steps_in, merge_df):
    result = training_init(merge_df, n_steps_in, model_daerah)
    return model_daerah, result


def main():
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
        result = process_model(model_daerah, n_steps_in, merge_df)
        results.append(result)

    model_dict = {model: result for model, result in results}
    print(model_dict)


if __name__ == "__main__":
    main()
