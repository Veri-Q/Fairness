FROM python:3.8.12-slim-buster

WORKDIR /home/qfairness

COPY . .

RUN pip install -r requirements.txt