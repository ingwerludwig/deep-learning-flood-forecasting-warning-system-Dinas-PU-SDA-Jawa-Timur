from flask import Flask, request, jsonify
from TimeSeriesModel import get_model
import pandas as pd
import json

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hello</h1>"

# Endpoint
# /api/predict

# Request Body
# {
#     "daerah" : "dhompo",
#     "model" : "lstm",
#     "data": [
#         {
#             "RC": 0.00,
#             "RL": 0.00,
#             "LP": 0.11,
#             "LD": 0.32,
#             "tanggal": "2022-09-20 20:00:00"
#         },
#         {
#             "RC": 0.00,
#             "RL": 0.00,
#             "LP": 0.11,
#             "LD": 0.31,
#             "tanggal": "2022-09-20 21:00:00"
#         },
#         {
#             "RC": 0.00,
#             "RL": 0.00,
#             "LP": 0.11,
#             "LD": 0.30,
#             "tanggal": "2022-09-20 22:00:00"
#         },
#         {
#             "RC": 0.00,
#             "RL": 0.00,
#             "LP": 0.11,
#             "LD": 0.29,
#             "tanggal": "2022-09-20 23:00:00"
#         },
#         {
#             "RC": 0.00,
#             "RL": 0.00,
#             "LP": 0.11,
#             "LD": 0.29,
#             "tanggal": "2022-09-21 00:00:00"
#         }
#     ]
# }
@app.route("/api/predict", methods=['POST'])
def api_pred():
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "Input data not provided"}), 400

    data_sensor = input_data.get('data', [])
    req_daerah = input_data.get('daerah')
    req_model = input_data.get('model', "lstm")

    daerah_model = f"{req_daerah}_{req_model}"
    select_model = get_model(daerah_model)

    if daerah_model == "dhompo_gru":
        n_steps_in = 3
    elif daerah_model == "dhompo_lstm":
        n_steps_in = 5
    elif daerah_model == "dhompo_tcn":
        n_steps_in = 5
    elif daerah_model == "purwodadi_gru":
        n_steps_in = 1
    elif daerah_model == "purwdoadi_lstm":
        n_steps_in = 2
    elif daerah_model == "purwodadi_tcn":
        n_steps_in = 2

    preprocessed_data = select_model.preprocess_data(pd.DataFrame(data_sensor), n_steps_in)
    prediction = select_model.model.predict(preprocessed_data, select_model.model.y_scaler)
    return prediction.to_json()


if __name__ == '__main__':
    app.run(host="localhost", port=8000)
