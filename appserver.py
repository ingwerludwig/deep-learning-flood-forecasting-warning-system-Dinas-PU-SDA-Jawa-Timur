from flask import Flask
from app.routes import api_bp
import warnings

warnings.filterwarnings("ignore")


def create_app():
    app = Flask('FFWS')
    app.register_blueprint(api_bp)
    return app
