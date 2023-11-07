from flask import Blueprint, request, jsonify
from datetime import timedelta
from TimeSeriesClass.timeseries import get_model
from utils.helpers import get_latest_date
import pandas as pd

api_bp = Blueprint('api', __name__)

@api_bp.route("/")
def home():
    return "<h1>Hello</h1>"

@api_bp.route("/api/predict", methods=['POST'])
def api_pred():
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "Input data not provided"}), 400

    data_sensor = input_data.get('data', [])

    arr_model_daerah = [
        "dhompo_gru",
        "dhompo_lstm",
        "purwodadi_gru",
        "purwodadi_lstm"
    ]

    model_dict = {item: {} for item in arr_model_daerah}

    latest_date = get_latest_date(data_sensor)

    for model_daerah in arr_model_daerah:
        select_model = get_model(model_daerah)

        if model_daerah == "dhompo_gru":
            n_steps_in = 3
            input_to_model = data_sensor[-3:]
        elif model_daerah == "dhompo_lstm":
            n_steps_in = 5
            input_to_model = data_sensor
        elif model_daerah == "dhompo_tcn":
            n_steps_in = 5
            input_to_model = data_sensor
        elif model_daerah == "purwodadi_gru":
            n_steps_in = 1
            input_to_model = data_sensor[-1:]
        elif model_daerah == "purwodadi_lstm":
            n_steps_in = 2
            input_to_model = data_sensor[-2:]
        elif model_daerah == "purwodadi_tcn":
            n_steps_in = 2
            input_to_model = data_sensor[-2:]

        new_date = latest_date + timedelta(hours=1)
        new_date_str = new_date.strftime("%Y-%m-%d %H:%M:%S")

        preprocessed_data = select_model.preprocess_data(pd.DataFrame(input_to_model), n_steps_in)
        prediction = select_model.model.predict(preprocessed_data, select_model.model.y_scaler)

        model_dict[model_daerah][new_date_str] = float(prediction[0])
    return jsonify(model_dict)
