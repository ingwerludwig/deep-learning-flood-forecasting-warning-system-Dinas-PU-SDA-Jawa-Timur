import pandas as pd
from database.db_utils import connect_db, close_db_connection, execute_sql_query
from sqlalchemy import text


def get_data_for_train():
    merge_df = pd.read_excel('Fix Data_Model Welang.xlsx')
    merge_df.drop('Unnamed: 0', axis=1, inplace=True)
    merge_df['DateTime'] = pd.to_datetime(merge_df['DateTime'])
    merge_df.sort_values(by='DateTime', ascending=True, inplace=True)
    merge_df.drop(['DateTime'], axis=1, inplace=True)
    return merge_df


def get_latest_rows():
    sql_query = "SELECT RC, RL, LP, LD, DateTime FROM (SELECT curah_hujan_cendono AS RC, curah_hujan_lawang AS RL, level_muka_air_purwodadi AS LP, level_muka_air_dhompo AS LD, tanggal AS DateTime FROM awlr_arr_per_jam ORDER BY tanggal DESC LIMIT 5) AS latest_data ORDER BY latest_data.DateTime ASC"

    connection = connect_db()
    result_data, result_column = execute_sql_query(connection, sql_query)

    df = pd.DataFrame(result_data, columns=result_column)

    decimal_columns = ['RC', 'RL', 'LP', 'LD']
    df[decimal_columns] = df[decimal_columns].astype(float)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')

    close_db_connection(connection)
    return df