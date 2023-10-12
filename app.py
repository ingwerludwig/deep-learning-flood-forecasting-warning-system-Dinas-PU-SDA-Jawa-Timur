from flask import Flask, request
from TimeSeriesModel import get_model

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hello</h1>"

# /api/predict?daerah=dhompo&model=lstm
@app.route("/api/predict")
def api_pred():
    req_daerah = request.args.get('daerah', type = str)
    req_model = request.args.get('model', default = "lstm", type = str)
    select_model = get_model(f"{req_daerah}_{req_model}")
    scaled_data ="" #Perform split_sequences and scaling
    result = select_model.predict(
        scaled_data,
        select_model.model,
        select_model.y_scaler
    )
    return result

if __name__ == '__main__':
    app.run(host="localhost", port=8000)