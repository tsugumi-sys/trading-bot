FROM python:3.8-slim-buster

RUN mkdir /app

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD [ "python", "app.py"]