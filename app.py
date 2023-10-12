from flask import Flask, request
from TimeSeriesModel import get_model
from preprocess_data import get_data

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hello</h1>"

# /api/predict?daerah=dhompo&model=lstm
@app.route("/api/predict")
def api_pred():
    req_daerah = request.args.get('daerah', type = str)
    req_model = request.args.get('model', default = "lstm", type = str)
    daerah_model = f"{req_daerah}_{req_model}"
    select_model = get_model(daerah_model)
    input_data = get_data()
    
    if(daerah_model == "dhompo_gru"):
        n_steps_in = 3
    elif(daerah_model == "dhompo_lstm"):
        n_steps_in = 5
    elif(daerah_model == "dhompo_tcn"):
        n_steps_in = 5
    elif(daerah_model == "purwodadi_gru"):
        n_steps_in = 1
    elif(daerah_model == "purwdoadi_lstm"):
        n_steps_in = 2
    elif(daerah_model == "purwodadi_tcn"):
        n_steps_in = 2
    
    preprocessed_data = select_model.preprocess_data(input_data, n_steps_in)
    prediction = select_model.predict(preprocessed_data, select_model.model.y_scaler)
    return prediction

if __name__ == '__main__':
    app.run(host="localhost", port=8000)