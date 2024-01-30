<p align="center">
        <a href="https://www.tensorflow.org/" target="_blank">
            <img src="https://www.tensorflow.org/images/tf_logo_social.png" width="200" alt="TensorFlow Logo">
        </a>
        &nbsp;&nbsp;&nbsp;
        <a href="https://flask.palletsprojects.com/" target="_blank">
            <!-- Replace the src attribute with the Flask logo URL -->
            <img src="https://miro.medium.com/v2/resize:fit:438/0*AZd8eYeNvupEXtRK.png" width="200" alt="Flask Logo">
        </a>
    </p>

<p align="center">
    <a href="https://github.com/tensorflow/tensorflow/actions">
        <img src="https://img.shields.io/github/workflow/status/tensorflow/tensorflow/CI" alt="Build Status">
    </a>
    <a href="https://pypi.org/project/tensorflow/"><img src="https://img.shields.io/pypi/dm/tensorflow" alt="Total Downloads"></a>
    <a href="https://pypi.org/project/tensorflow/"><img src="https://img.shields.io/pypi/v/tensorflow" alt="Latest Stable Version"></a>
    <a href="https://pypi.org/project/tensorflow/"><img src="https://img.shields.io/pypi/l/tensorflow" alt="License"></a>
</p>

## About Repository

Applied Deep Learning project Repository for Flood Forecasting Warning Sistem Dinas PU Sumber Daya Air Jatim <br>

## Tech Stack For Flood Forecasting Warning System Web Monitoring
| Web Development                                    | Artificial Intelligence                         |
|----------------------------------------------------|-------------------------------------------------|
| [![Laravel][Laravel.com]][Laravel-url]             | [![Python][python.com]][python-url]             |
| [![MySQL][mysql.com]][mysql-url]                   | [![Tensorflow][TensorFlow.com]][TensorFlow-url] |
| [![React][React.com]][React-url]                   | [![Flask][Flask.com]][Flask-url]                |
| [![TailwindCSS][TailwindCSS.com]][TailwindCSS-url] | [![Keras][Keras.com]][Keras-url]                |

## Requirement for Applied AI (Deep Learning)
* [Python 3.9 or above](https://www.python.org)
* [Pip](https://pypi.org/project/pip/)

## Getting Started
1. Clone this repository
```
git clone https://github.com/ingwerludwig/deep-learning-flood-forecasting-warning-system-Dinas-PU-SDA-Jawa-Timur.git
```
2. Install All Dependencies <br>
* If you Windows/Linux User
```
pip install -r requirements.txt
```
* If you MacOS User
```
pip install -r requirements-macos.txt
```
3. Copy the .env.example to .env
```
cp .env.example .env
```
4. Adjust your .env with your environment match with [Backend Database environment](https://github.com/ingwerludwig/web-flood-forecasting-warning-system-Dinas-PU-SDA-Jatim/blob/master/.env.example) if you want to integrate the Backend and Deep Learning
5. Install the pm2 to auto restarting Flask if the aplication going down
```
npm install pm2 -g
```
6. Start the Flask Application
* Adjust 127.0.0.0 (Host) and 8000 (Port) as your needs
```
pm2 start "gunicorn --bind 127.0.0.1:8000 wsgi:gunicorn_app"
```
## Starting with Dockerized Python Flask (optional)
7. Pull Docker Image
```
docker pull ingwerludwig/dinas-pu-ffws-deep-learning:v1
```
8. Import the docker-compose.yml
9. Run the Docker Container in docker-compose.yml working directory
```
docker compose up -d
```

## Web Endpoint Documentation
https://www.postman.com/myprivatepersonal/workspace/ffws-api-endpoint-testing/collection/26715144-6bb171e5-e1f5-4df6-842c-b0e6417eb53f?action=share&creator=26715144


[Laravel.com]: https://img.shields.io/badge/laravel-%23FF2D20.svg?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[mysql.com]: https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white
[mysql-url]: https://laravel.com](https://www.mysql.com)https://www.mysql.com
[python.com]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org
[TensorFlow.com]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org
[React.com]: https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB
[React-url]: https://react.dev
[Flask.com]: https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/en/3.0.x/
[TailwindCSS.com]: https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white
[TailwindCSS-url]: https://tailwindcss.com
[Keras.com]: https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white
[Keras-url]: https://keras.io/api/
