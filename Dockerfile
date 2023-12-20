FROM python:3.9
RUN apt-get install -y pkg-config

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "gunicorn", "--bind" , "0.0.0.0:8000", "wsgi:gunicorn_app"]