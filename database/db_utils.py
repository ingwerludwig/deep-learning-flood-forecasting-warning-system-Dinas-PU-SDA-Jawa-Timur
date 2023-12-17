import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
db_url = f"mysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


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
