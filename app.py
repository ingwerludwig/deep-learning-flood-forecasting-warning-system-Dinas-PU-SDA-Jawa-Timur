from flask import Flask, request
import pandas as pd
from model import get_model
from model_prediction import model_predict

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hello</h1>"

# /api/predict?daerah=dhompo&model=lstm
@app.route("/api/predict")
def api_pred():
    req_daerah = request.args.get('daerah', type = str)
    req_model = request.args.get('model', default = "lstm", type = str)
    model, x_scaler, y_scaler = get_model(f"{req_daerah}_{req_model}")
    scaled_data ="" #Perform split_sequences and scaling
    result = model_predict(scaled_data,model,y_scaler)
    return result

if __name__ == '__main__':
    app.run(host="localhost", port=8000)