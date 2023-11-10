from sklearn.model_selection import train_test_split


def train_test_split_data(merge_df):
    train_percent = 0.7
    validation_percent = 0.2

    new_merge_df = merge_df
    train_size = int(len(new_merge_df) * train_percent)
    validation_size = int(len(new_merge_df) * validation_percent)

    df_train = new_merge_df[:train_size]
    df_validation = new_merge_df[train_size:(train_size + validation_size)]
    df_test = new_merge_df[train_size + validation_size:]

    return df_train, df_validation, df_test
