import pandas as pd
from database.db_utils import close_db_connection, connect_db, execute_sql_query


def get_data_for_train():
    sql_query = "SELECT RC, RL, LP, LD, DateTime FROM (SELECT curah_hujan_cendono AS RC, curah_hujan_lawang AS RL, level_muka_air_purwodadi AS LP, level_muka_air_dhompo AS LD, tanggal AS DateTime FROM awlr_arr_per_jam ORDER BY tanggal DESC LIMIT 720) AS latest_data ORDER BY latest_data.DateTime ASC"

    connection = connect_db()
    result_data, result_column = execute_sql_query(connection, sql_query)

    df = pd.DataFrame(result_data, columns=result_column)

    decimal_columns = ['RC', 'RL', 'LP', 'LD']
    df[decimal_columns] = df[decimal_columns].astype(float)
    df.drop('DateTime',axis=1, inplace=True)

    close_db_connection(connection)
    return df


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
