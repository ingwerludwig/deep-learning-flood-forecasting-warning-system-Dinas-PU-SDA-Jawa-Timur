import pandas as pd

def model_predict(scaled_data,model,y_scaler):
    predictions = y_scaler.inverse_transform(model.predict_generator(scaled_data).reshape(-1,1))
    predictions = pd.DataFrame(predictions)
    predictions = predictions
    predictions = pd.DataFrame(predictions)
    result = predictions[0]
    return result