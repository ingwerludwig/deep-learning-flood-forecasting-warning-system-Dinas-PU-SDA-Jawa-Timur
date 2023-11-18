from sqlalchemy import create_engine, text

db_url = 'mysql://root@127.0.0.1:3306/ffws'


def connect_db():
    engine = create_engine(db_url)
    connection = engine.connect()
    return connection


def execute_sql_query(connection, sql_query):
    try:
        result = connection.execute(text(sql_query))
        return result.fetchall(), result.keys()
    except Exception as e:
        return e


def close_db_connection(connection):
    connection.close()
