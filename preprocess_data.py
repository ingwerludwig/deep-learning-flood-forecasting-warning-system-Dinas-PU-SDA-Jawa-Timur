import pandas as pd
import array

def get_data():
    data = [[0.0, 0.0, 0.071, 0.196],
            [0.0, 0.0, 0.071, 0.183],
            [0.0, 0.0, 0.068, 0.164],
            [0.0, 0.0, 0.061, 0.138],
            [0.0, 0.0, 0.049, 0.131]]
    df_test = pd.DataFrame(data, columns=["RC", "RL", "LP", "LD"])
    return df_test

def split_sequences_sliding(sequences, n_steps_in, n_steps_out, step=1):
    X, y = list(), list()
    for i in range(0, len(sequences) - n_steps_in - n_steps_out + 1, step):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)