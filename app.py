from flask import Flask
from app.routes import api_bp
import atexit
from database.db_utils import connect_db, close_db_connection

app = Flask(__name__)

connection = connect_db()
atexit.register(close_db_connection, connection)

app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(host="localhost", port=8000)
