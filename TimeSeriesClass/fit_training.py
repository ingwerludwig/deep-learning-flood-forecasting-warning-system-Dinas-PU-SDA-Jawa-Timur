import pickle


def train_data(model, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_scaler, y_scaler, dearah_model):
    try:
        epoch = 50

        model.compile(optimizer='adam', loss='mae', metrics='mae')
        model.fit(x_train_scaled, y_train_scaled, batch_size=64, epochs=epoch,
                  validation_data=(x_val_scaled, y_val_scaled))

        model.save(f"{dearah_model}.h5")

        with open(f"{dearah_model}_x_scaler.pkl", 'wb') as file:
            pickle.dump(x_scaler, file)

        with open(f"{dearah_model}_y_scaler.pkl", 'wb') as file:
            pickle.dump(y_scaler, file)

        return "successful"
    except Exception as e:
        print(f"Error during training: {e}")
        return "failed"
