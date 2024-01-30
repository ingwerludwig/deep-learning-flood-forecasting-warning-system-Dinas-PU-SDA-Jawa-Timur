FROM python:3.9 as build
RUN pip install mysqlclient

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim
RUN apt-get update && apt-get install -y libmariadb3

RUN pip install gunicorn

RUN useradd -m -r -s /bin/bash jatim

COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

USER jatim
RUN mkdir /home/jatim/code
WORKDIR /home/jatim/code
COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "wsgi:gunicorn_app"]