from tensorflow.keras.models import load_model
import pickle

# Dhompo GRU
with open("", "rb") as file:
    dhompo_gru_x_scaler = pickle.load("")

with open("", "rb") as file:
    dhompo_gru_y_scaler = pickle.load("")
    
# Dhompo LSTM
with open("", "rb") as file:
    dhompo_lstm_x_scaler = pickle.load("")

with open("", "rb") as file:
    dhompo_lstm_y_scaler = pickle.load("")

# Dhompo TCN
with open("", "rb") as file:
    dhompo_tcn_x_scaler = pickle.load("")

with open("", "rb") as file:
    dhompo_tcn_y_scaler = pickle.load("")

# Purwodadi GRU
with open("", "rb") as file:
    purwodadi_gru_x_scaler = pickle.load("")
    
with open("", "rb") as file:
    purwodadi_gru_y_scaler = pickle.load("")
    
# Purwodadi LSTM
with open("", "rb") as file:
    purwodadi_lstm_x_scaler = pickle.load("")
    
with open("", "rb") as file:
    purwodadi_lstm_y_scaler = pickle.load("")
    
# Purwodadi LSTM
with open("", "rb") as file:
    purwodadi_tcn_x_scaler = pickle.load("")
    
with open("", "rb") as file:
    purwodadi_tcn_y_scaler = pickle.load("")

model_dhompo_gru = load_model("")
model_dhompo_lstm = load_model("")
model_dhompo_tcn = load_model("")
model_purwodadi_gru = load_model("")
model_purwodadi_lstm = load_model("") 
model_purwodadi_tcn = load_model("")

def dhompo_gru():
    model = model_dhompo_gru
    x_scaler = dhompo_gru_x_scaler
    y_scaler = dhompo_gru_y_scaler
    return model,x_scaler,y_scaler

def dhompo_lstm():
    model = model_dhompo_lstm
    x_scaler = dhompo_lstm_x_scaler
    y_scaler = dhompo_lstm_y_scaler
    return model, x_scaler, y_scaler

def dhompo_tcn():
    model = model_dhompo_tcn
    x_scaler = dhompo_tcn_x_scaler
    y_scaler = dhompo_tcn_y_scaler
    return model, x_scaler, y_scaler

def purwodadi_gru():
    model = model_purwodadi_gru
    x_scaler = purwodadi_gru_x_scaler
    y_scaler = purwodadi_gru_y_scaler
    return model, x_scaler, y_scaler

def purwodadi_lstm():
    model = model_purwodadi_lstm
    x_scaler = purwodadi_lstm_x_scaler
    y_scaler = purwodadi_lstm_y_scaler
    return model,x_scaler,y_scaler

def purwodadi_tcn():
    model = model_purwodadi_tcn
    x_scaler = purwodadi_tcn_x_scaler
    y_scaler = purwodadi_tcn_y_scaler
    return model,x_scaler,y_scaler

def get_model(model_name):
    cases = {
        "dhompo_gru": dhompo_gru(),
        "dhompo_lstm": dhompo_lstm(),
        "dhompo_tcn": dhompo_tcn(),
        "purwodadi_gru": purwodadi_gru(),
        "purwodadi_lstm": purwodadi_lstm(),
        "purwodadi_tcn": purwodadi_tcn()
    }
    model, x_scaler, y_scaler = cases.get(model_name)()
    return model, x_scaler, y_scaler