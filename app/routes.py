from flask import Blueprint, jsonify
from datetime import timedelta
from TimeSeriesClass.timeseries import get_model
from TimeSeriesClass.get_data import get_latest_rows
import pandas as pd

api_bp = Blueprint('api', __name__)


@api_bp.route("/")
def home():
    return "<h1>Hello</h1>"


@api_bp.route("/health")
def health_check():
    return "OK"


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


@api_bp.route("/api/predict", methods=['POST'])
def api_pred():
    arr_model_daerah = [
        # "dhompo_gru",
        "dhompo_lstm",
        # "dhompo_tcn",
        "purwodadi_gru",
        # "purwodadi_lstm",
        # "purwodadi_tcn"
    ]

    model_dict = {item: {'predictions': {}} for item in arr_model_daerah}

    for model_daerah in arr_model_daerah:
        select_model = get_model(model_daerah)
        n_steps_in = get_n_steps_in_for_model(model_daerah)
        n_steps_out = get_n_steps_out_for_model(model_daerah)
        input_to_model = get_latest_rows()
        input_to_model = input_to_model.tail(n_steps_in)

        date = input_to_model.tail(1).iloc[0]['DateTime']
        predicted_from_time = date.strftime("%Y-%m-%d %H:%M:%S")

        preprocessed_data = select_model.preprocess_data(pd.DataFrame(input_to_model), n_steps_in)
        prediction = select_model.model.predict(preprocessed_data, select_model.model.y_scaler, n_steps_out)

        index_looping = 1

        prediction_values = prediction.values.flatten()

        prediction_dict = {
            (date + timedelta(hours=i + index_looping)).strftime("%Y-%m-%d %H:%M:%S"): {
                'value': float(value)
            } for i, value in enumerate(prediction_values)
        }

        predicted_for_time = (date + timedelta(hours=n_steps_out)).strftime("%Y-%m-%d %H:%M:%S")
        model_dict[model_daerah]['predicted_until_time'] = predicted_for_time
        model_dict[model_daerah]['predicted_from_time'] = predicted_from_time
        model_dict[model_daerah]['predictions'] = prediction_dict

    return jsonify(model_dict)
