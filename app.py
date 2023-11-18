from flask import Flask
from app.routes import api_bp
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

app.register_blueprint(api_bp)


if __name__ == '__main__':
    app.run(host="192.168.18.51", port=8000)
